import torch
import tqdm
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
import matplotlib


def colorize_depth(depth, cmap="Spectral"):
    # colorize
    cm = matplotlib.colormaps[cmap]
    # (B, N, H, W, 3)
    depth_colored = cm(depth, bytes=False)[..., 0:3]  # value from 0 to 1
    return depth_colored


class DAVOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    disparity: np.ndarray
    disparity_colored: np.ndarray
    image: np.ndarray


class DAVPipeline(DiffusionPipeline):
    def __init__(self, vae, unet, unet_interp, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae, unet=unet, unet_interp=unet_interp, scheduler=scheduler
        )

    def encode(self, input):
        num_frames = input.shape[1]
        input = input.flatten(0, 1)
        latent = self.vae.encode(input.to(self.vae.dtype)).latent_dist.mode()
        latent = latent * self.vae.config.scaling_factor
        latent = latent.reshape(-1, num_frames, *latent.shape[1:])
        return latent

    def decode(self, latents, decode_chunk_size=16):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        num_frames = latents.shape[1]
        latents = latents.flatten(0, 1)
        latents = latents / self.vae.config.scaling_factor

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            frame = self.vae.decode(
                latents[i : i + decode_chunk_size].to(self.vae.dtype),
                num_frames=num_frames_in,
            ).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch, frames, channels, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:])
        return frames.to(torch.float32)

    def single_infer(self, rgb, position_ids=None, num_inference_steps=None):
        rgb_latent = self.encode(rgb)
        noise_latent = torch.randn_like(rgb_latent)

        self.scheduler.set_timesteps(num_inference_steps, device=rgb.device)
        timesteps = self.scheduler.timesteps

        image_embeddings = torch.zeros((noise_latent.shape[0], 1, 1024)).to(
            noise_latent
        )

        for i, t in enumerate(timesteps):
            latent_model_input = noise_latent

            latent_model_input = torch.cat([latent_model_input, rgb_latent], dim=2)

            # [batch_size, num_frame, 4, h, w]
            model_output = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                position_ids=position_ids,
            ).sample

            # compute the previous noisy sample x_t -> x_t-1
            noise_latent = self.scheduler.step(
                model_output, t, noise_latent
            ).prev_sample

        return noise_latent

    def single_interp_infer(
        self, rgb, masked_depth_latent, mask, num_inference_steps=None
    ):
        rgb_latent = self.encode(rgb)
        noise_latent = torch.randn_like(rgb_latent)

        self.scheduler.set_timesteps(num_inference_steps, device=rgb.device)
        timesteps = self.scheduler.timesteps

        image_embeddings = torch.zeros((noise_latent.shape[0], 1, 1024)).to(
            noise_latent
        )

        for i, t in enumerate(timesteps):
            latent_model_input = noise_latent

            latent_model_input = torch.cat(
                [latent_model_input, rgb_latent, masked_depth_latent, mask], dim=2
            )

            # [batch_size, num_frame, 4, h, w]
            model_output = self.unet_interp(
                latent_model_input, t, encoder_hidden_states=image_embeddings
            ).sample

            # compute the previous noisy sample x_t -> x_t-1
            noise_latent = self.scheduler.step(
                model_output, t, noise_latent
            ).prev_sample

        return noise_latent

    def __call__(
        self,
        image,
        num_frames,
        num_overlap_frames,
        num_interp_frames,
        decode_chunk_size,
        num_inference_steps,
    ):
        self.vae.to(dtype=torch.float16)

        # (1, N, 3, H, W)
        image = image.unsqueeze(0)
        B, N = image.shape[:2]
        rgb = image * 2 - 1  # [-1, 1]

        if N <= num_frames or N <= num_interp_frames + 2 - num_overlap_frames:
            depth_latent = self.single_infer(
                rgb, num_inference_steps=num_inference_steps
            )
        else:
            assert 2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
            assert num_frames % 2 == 0

            key_frame_indices = []
            for i in range(0, N, num_interp_frames + 2 - num_overlap_frames):
                if (
                    i + num_interp_frames + 1 >= N
                    or len(key_frame_indices) >= num_frames
                ):
                    break
                key_frame_indices.append(i)
                key_frame_indices.append(i + num_interp_frames + 1)

            key_frame_indices = torch.tensor(key_frame_indices, device=rgb.device)

            sorted_key_frame_indices, origin_indices = torch.sort(key_frame_indices)
            key_rgb = rgb[:, sorted_key_frame_indices]
            key_depth_latent = self.single_infer(
                key_rgb,
                sorted_key_frame_indices.unsqueeze(0).repeat(B, 1),
                num_inference_steps=num_inference_steps,
            )
            key_depth_latent = key_depth_latent[:, origin_indices]

            torch.cuda.empty_cache()

            depth_latent = []
            pre_latent = None
            for i in tqdm.tqdm(range(0, len(key_frame_indices), 2)):
                frame1 = key_depth_latent[:, i]
                frame2 = key_depth_latent[:, i + 1]
                masked_depth_latent = torch.zeros(
                    (B, num_interp_frames + 2, *key_depth_latent.shape[2:])
                ).to(key_depth_latent)
                masked_depth_latent[:, 0] = frame1
                masked_depth_latent[:, -1] = frame2

                mask = torch.zeros_like(masked_depth_latent)
                mask[:, [0, -1]] = 1.0

                latent = self.single_interp_infer(
                    rgb[:, key_frame_indices[i] : key_frame_indices[i + 1] + 1],
                    masked_depth_latent,
                    mask,
                    num_inference_steps=num_inference_steps,
                )
                latent = latent[:, 1:-1]

                if pre_latent is not None:
                    overlap_a = pre_latent[
                        :, pre_latent.shape[1] - (num_overlap_frames - 2) :
                    ]
                    overlap_b = latent[:, : (num_overlap_frames - 2)]
                    ratio = (
                        torch.linspace(0, 1, num_overlap_frames - 2)
                        .to(overlap_a)
                        .view(1, -1, 1, 1, 1)
                    )
                    overlap = overlap_a * (1 - ratio) + overlap_b * ratio
                    pre_latent[:, pre_latent.shape[1] - (num_overlap_frames - 2) :] = (
                        overlap
                    )
                    depth_latent.append(pre_latent)

                pre_latent = latent[:, (num_overlap_frames - 2) if i > 0 else 0 :]

                torch.cuda.empty_cache()

            depth_latent.append(pre_latent)
            depth_latent = torch.cat(depth_latent, dim=1)

            # dicard the first and last key frames
            image = image[:, key_frame_indices[0] + 1 : key_frame_indices[-1]]
            assert depth_latent.shape[1] == image.shape[1]

        disparity = self.decode(depth_latent, decode_chunk_size=decode_chunk_size)
        disparity = disparity.mean(dim=2, keepdim=False)
        disparity = torch.clamp(disparity * 0.5 + 0.5, 0.0, 1.0)

        # (N, H, W)
        disparity = disparity.squeeze(0)
        # (N, H, W, 3)
        mid_d, max_d = disparity.min(), disparity.max()
        disparity_colored = torch.clamp((max_d - disparity) / (max_d - mid_d), 0.0, 1.0)
        disparity_colored = colorize_depth(disparity_colored.cpu().numpy())
        disparity_colored = (disparity_colored * 255).astype(np.uint8)
        image = image.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        disparity = disparity.cpu().numpy()

        return DAVOutput(
            disparity=disparity,
            disparity_colored=disparity_colored,
            image=image,
        )
