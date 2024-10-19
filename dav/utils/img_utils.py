from PIL import Image
import cv2
import numpy as np
import os
import tempfile


def resize(img, size):
    assert img.dtype == np.uint8
    pil_image = Image.fromarray(img)
    pil_image = pil_image.resize(size, Image.LANCZOS)
    resized_img = np.array(pil_image)
    return resized_img


def crop(img, start_h, start_w, crop_h, crop_w):
    img_src = np.zeros((crop_h, crop_w, *img.shape[2:]), dtype=img.dtype)
    hsize, wsize = crop_h, crop_w
    dh, dw, sh, sw = start_h, start_w, 0, 0
    if dh < 0:
        sh = -dh
        hsize += dh
        dh = 0
    if dh + hsize > img.shape[0]:
        hsize = img.shape[0] - dh
    if dw < 0:
        sw = -dw
        wsize += dw
        dw = 0
    if dw + wsize > img.shape[1]:
        wsize = img.shape[1] - dw
    img_src[sh : sh + hsize, sw : sw + wsize] = img[dh : dh + hsize, dw : dw + wsize]
    return img_src


def imresize_max(img, size, min_side=False):
    new_img = []
    for i, _img in enumerate(img):
        h, w = _img.shape[:2]
        ori_size = min(h, w) if min_side else max(h, w)
        _resize = min(size / ori_size, 1.0)
        new_size = (int(w * _resize), int(h * _resize))
        new_img.append(resize(_img, new_size))
    return new_img


def imcrop_multi(img, multiple=32):
    new_img = []
    for i, _img in enumerate(img):
        crop_size = (
            _img.shape[0] // multiple * multiple,
            _img.shape[1] // multiple * multiple,
        )
        start_h = int(0.5 * max(0, _img.shape[0] - crop_size[0]))
        start_w = int(0.5 * max(0, _img.shape[1] - crop_size[1]))
        _img_src = crop(_img, start_h, start_w, crop_size[0], crop_size[1])
        new_img.append(_img_src)
    return new_img


def read_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    # (N, H, W, 3)
    return frames, fps


def read_image(image_path):
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # (1, H, W, 3)
    return [frame]


def write_video(video_path, frames, fps):
    tmp_dir = os.path.join(os.path.dirname(video_path), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        write_image(os.path.join(tmp_dir, f"{i:06d}.png"), frame)
    # it will cause visual compression artifacts
    ffmpeg_command = [
        "ffmpeg",
        "-f",
        "image2",
        "-framerate",
        f"{fps}",
        "-i",
        os.path.join(tmp_dir, "%06d.png"),
        "-b:v",
        "5626k",
        "-y",
        video_path,
    ]
    os.system(" ".join(ffmpeg_command))
    os.system(f"rm -rf {tmp_dir}")


def write_image(image_path, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, frame)
