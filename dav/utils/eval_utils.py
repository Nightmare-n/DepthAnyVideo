import time

import numpy as np
import torch
import tqdm
import os
import matplotlib
import cv2


def d1(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    return d1


def d2(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d2 = (thresh < 1.25**2).mean()
    return d2


def d3(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d3 = (thresh < 1.25**3).mean()
    return d3


def rmse(gt, pred):
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    return rmse


def rmse_log(gt, pred):
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    return rmse_log


def abs_rel(gt, pred):
    abs_rel = (np.abs(gt - pred) / gt).mean()
    return abs_rel


def sq_rel(gt, pred):
    sq_rel = (((gt - pred) ** 2) / gt).mean()
    return sq_rel


def log10(gt, pred):
    log10 = np.abs(np.log10(pred) - np.log10(gt)).mean()
    return log10


def silog(gt, pred):
    # https://guillesanbri.com/Scale-Invariant-Loss/
    err = np.log(pred) - np.log(gt)
    silog = 100 * np.sqrt(np.mean(err**2) - np.mean(err) ** 2)
    return silog


def depth2point(depth, mask, img2lidar):
    h, w = depth.shape
    ys, xs = np.meshgrid(
        np.linspace(0.5, h - 0.5, h),
        np.linspace(0.5, w - 0.5, w),
        indexing="ij",
    )
    # (H, W, 4)
    points = np.stack([xs, ys, depth, np.ones_like(xs)], axis=-1)
    points = points[mask]
    points[..., :2] *= points[..., 2:3]
    points = points @ img2lidar.T
    points = points[..., :3]
    return points


def point2depth(points, warp_mask, warp_img2lidar):
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    lidar2img = np.linalg.inv(warp_img2lidar)
    points = points @ lidar2img.T
    depth = points[..., 2]
    eps = 1e-6
    mask = depth > eps
    cam_points = points[..., :2] / np.clip(points[..., 2:3], a_min=eps, a_max=None)
    cam_coords = np.round(cam_points).astype(np.int32)
    h, w = warp_mask.shape
    mask &= (
        (cam_coords[..., 0] >= 0)
        & (cam_coords[..., 0] < w)
        & (cam_coords[..., 1] >= 0)
        & (cam_coords[..., 1] < h)
    )
    depth = depth[mask]
    cam_coords = cam_coords[mask]
    warp_depth = np.zeros((h, w), dtype=np.float32)
    warp_depth[cam_coords[..., 1], cam_coords[..., 0]] = depth
    warp_depth = warp_depth * warp_mask
    return warp_depth


def tae(
    depth_pred_a,
    mask_a,
    img2lidar_a,
    depth_pred_b,
    mask_b,
    img2lidar_b,
):
    depth_a2b = point2depth(
        depth2point(depth_pred_a, mask_a, img2lidar_a), mask_b, img2lidar_b
    )
    mask = (depth_a2b > 1e-6) & mask_b
    error_a2b = abs_rel(depth_pred_b[mask], depth_a2b[mask])
    depth_b2a = point2depth(
        depth2point(depth_pred_b, mask_b, img2lidar_b), mask_a, img2lidar_a
    )
    mask = (depth_b2a > 1e-6) & mask_a
    error_b2a = abs_rel(depth_pred_a[mask], depth_b2a[mask])
    return 0.5 * (error_a2b + error_b2a)


def tas(
    depth_pred_a,
    mask_a,
    img2lidar_a,
    depth_pred_b,
    mask_b,
    img2lidar_b,
):
    depth_a2b = point2depth(
        depth2point(depth_pred_a, mask_a, img2lidar_a), mask_b, img2lidar_b
    )
    mask = (depth_a2b > 1e-6) & mask_b
    score_a2b = d1(depth_pred_b[mask], depth_a2b[mask])
    depth_b2a = point2depth(
        depth2point(depth_pred_b, mask_b, img2lidar_b), mask_a, img2lidar_a
    )
    mask = (depth_b2a > 1e-6) & mask_a
    score_b2a = d1(depth_pred_a[mask], depth_b2a[mask])
    return 0.5 * (score_a2b + score_b2a)


def colorize_depth(depth, min_depth, max_depth, cmap="Spectral"):
    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    # (B, N, H, W, 3)
    depth_colored = cm(depth, bytes=False)[..., 0:3]  # value from 0 to 1
    return depth_colored


def batch_eval_func(pred_dict, result_dir, dataset_name, eval_cfg):
    metric = {
        "num_sample": 0,
    }
    for key in eval_cfg.eval_metric:
        metric[key] = 0

    batch_pred = pred_dict["disparity_pred"]
    batch_img = pred_dict["img"]

    # (B, N, H, W)
    batch_gt = pred_dict["depth"]
    batch_mask = pred_dict["depth_mask"] > 1e-6
    if "lidar2cam" not in pred_dict.keys():
        img2lidar = None
    else:
        lidar2cam = pred_dict["lidar2cam"]
        cam2img = pred_dict["cam2img"]
        img2lidar = torch.inverse(cam2img @ lidar2cam).cpu().numpy()

    if eval_cfg[dataset_name].get("eval_region", None) is not None:
        eval_mask = torch.zeros_like(batch_gt)
        t, b, l, r = eval_cfg[dataset_name].eval_region
        eval_mask[..., t:b, l:r] = 1.0
        batch_mask &= eval_mask > 1e-6

    assert batch_gt.shape == batch_pred.shape == batch_mask.shape

    for b_idx, (gt, pred, mask) in enumerate(zip(batch_gt, batch_pred, batch_mask)):
        # (N, H, W)
        fid = torch.stack([x.new_ones(x.shape) * i for i, x in enumerate(gt)]).long()
        gt_masked, pred_masked = gt[mask].cpu().numpy(), pred[mask].cpu().numpy()
        fid_masked = fid[mask].cpu().numpy()
        if gt_masked.shape[0] == 0:
            continue
        pred_scaled = pred.clone().cpu().numpy()

        if eval_cfg.fit_scale_shift:
            if not eval_cfg.get("temporal_fit", False):
                for f_idx in range(mask.shape[0]):
                    igt, ipred = (
                        gt_masked[fid_masked == f_idx][:, None],
                        pred_masked[fid_masked == f_idx][:, None],
                    )
                    A = np.concatenate([ipred, np.ones_like(ipred)], axis=-1)
                    X = np.linalg.lstsq(
                        A, 1 / np.clip(igt, a_min=1e-6, a_max=None), rcond=None
                    )[0]
                    scale, shift = X
                    pred_masked[fid_masked == f_idx] = (
                        pred_masked[fid_masked == f_idx] * scale + shift
                    )
                    pred_scaled[f_idx] = pred_scaled[f_idx] * scale + shift
            else:
                pred_masked = pred_masked.reshape((-1, 1))
                gt_masked = gt_masked.reshape((-1, 1))
                A = np.concatenate([pred_masked, np.ones_like(pred_masked)], axis=-1)
                X = np.linalg.lstsq(
                    A, 1 / np.clip(gt_masked, a_min=1e-6, a_max=None), rcond=None
                )[0]
                scale, shift = X
                pred_masked = pred_masked * scale + shift
                pred_scaled = pred_scaled * scale + shift

        # disparity to depth
        pred_masked = 1 / np.clip(pred_masked, a_min=1e-6, a_max=None)
        pred_masked = np.clip(
            pred_masked,
            a_min=eval_cfg[dataset_name].depth_range[0],
            a_max=eval_cfg[dataset_name].depth_range[1],
        )
        pred_scaled = 1 / np.clip(pred_scaled, a_min=1e-6, a_max=None)
        pred_scaled = np.clip(
            pred_scaled,
            a_min=eval_cfg[dataset_name].depth_range[0],
            a_max=eval_cfg[dataset_name].depth_range[1],
        )

        temporal_metric = ["tae", "tas"]
        for key in eval_cfg.eval_metric:
            if key not in temporal_metric or img2lidar is None:
                continue
            _tsim = 0.0
            for f_idx in range(mask.shape[0] - 1):
                tsim = globals()[key](
                    pred_scaled[f_idx],
                    mask[f_idx].cpu().numpy(),
                    img2lidar[b_idx, f_idx],
                    pred_scaled[f_idx + 1],
                    mask[f_idx + 1].cpu().numpy(),
                    img2lidar[b_idx, f_idx + 1],
                )
                _tsim += tsim
            metric[key] += _tsim / (mask.shape[0] - 1) * mask.shape[0]

        for f_idx in range(mask.shape[0]):
            igt, ipred = (
                gt_masked[fid_masked == f_idx],
                pred_masked[fid_masked == f_idx],
            )
            for key in eval_cfg.eval_metric:
                if key in temporal_metric:
                    continue
                val = globals()[key](igt, ipred)
                metric[key] += val
            metric["num_sample"] += 1

    return metric
