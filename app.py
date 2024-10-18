import gradio as gr
import logging
import os
import random
import tempfile
import time

from easydict import EasyDict
import numpy as np
import torch
from dav.pipelines import DAVPipeline
from dav.models import UNetSpatioTemporalRopeConditionModel
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from dav.utils import img_utils

def seed_all(seed: int = 0):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load models once to avoid reloading on every inference
def load_models(model_base, device):
    vae = AutoencoderKLTemporalDecoder.from_pretrained(model_base, subfolder="vae")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_base, subfolder="scheduler"
    )
    unet = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        model_base, subfolder="unet"
    )
    unet_interp = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        model_base, subfolder="unet_interp"
    )
    pipe = DAVPipeline(
        vae=vae,
        unet=unet,
        unet_interp=unet_interp,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)
    return pipe

# Load models at startup
MODEL_BASE = "hhyangcs/depth-any-video"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = load_models(MODEL_BASE, DEVICE)
logging.info(f"Models loaded on {DEVICE}")

def depth_any_video(
    file,
    denoise_steps=3,
    num_frames=32,
    decode_chunk_size=16,
    num_interp_frames=16,
    num_overlap_frames=6,
    max_resolution=1024,
    seed=None
):
    """
    Perform depth estimation on the uploaded video/image.
    """
    with open(file, "rb") as f1:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the uploaded file
            input_path = os.path.join(tmp_dir, file.name)
            with open(input_path, "wb") as f:
                f.write(f1.read())

            # Set up output directory
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Prepare configuration
            cfg = EasyDict({
                "model_base": MODEL_BASE,
                "data_dir": "./5.mp4", # str(file),# input_path,
                "output_dir": output_dir,
                "denoise_steps": denoise_steps,
                "num_frames": num_frames,
                "decode_chunk_size": decode_chunk_size,
                "num_interp_frames": num_interp_frames,
                "num_overlap_frames": num_overlap_frames,
                "max_resolution": max_resolution,
                "seed": seed if seed is not None else int(time.time())
            })

            seed_all(cfg.seed)

            file_name = os.path.splitext(os.path.basename(cfg.data_dir))[0]
            is_video = cfg.data_dir.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

            if is_video:
                num_interp_frames = cfg.num_interp_frames
                num_overlap_frames = cfg.num_overlap_frames
                num_frames = cfg.num_frames
                assert num_frames % 2 == 0, "num_frames should be even."
                assert (
                    2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
                ), "Invalid frame overlap."
                max_frames = (num_interp_frames + 2 - num_overlap_frames) * (num_frames // 2)
                image, fps = img_utils.read_video(cfg.data_dir, max_frames=max_frames)
            else:
                image = img_utils.read_image(cfg.data_dir)

            image = img_utils.imresize_max(image, cfg.max_resolution)
            image = img_utils.imcrop_multi(image)
            image_tensor = [_img / 255.0 for _img in image]
            image_tensor = [torch.from_numpy(_img).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE) for _img in image_tensor]
            # image_tensor = [torch.from_numpy(_img).permute(3, 1, 2).to(DEVICE) for _img in image_tensor]
            # Stack tensors along a new dimension (batch dimension)
            image_tensor = torch.stack(image_tensor)
            with torch.no_grad():
                pipe_out = pipe(
                    image_tensor,
                    num_frames=cfg.num_frames,
                    num_overlap_frames=cfg.num_overlap_frames,
                    num_interp_frames=cfg.num_interp_frames,
                    decode_chunk_size=cfg.decode_chunk_size,
                )

            disparity = pipe_out.disparity
            disparity_colored = pipe_out.disparity_colored
            image = pipe_out.image
            # (N, H, 2 * W, 3)
            merged = np.concatenate(
                [
                    image,
                    disparity_colored,
                ],
                axis=2,
            )

            if is_video:
                output_path = os.path.join(cfg.output_dir, f"{file_name}_depth.mp4")
                img_utils.write_video(
                    output_path,
                    merged,
                    fps,
                )
                return output_path
            else:
                output_path = os.path.join(cfg.output_dir, f"{file_name}_depth.png")
                img_utils.write_image(
                    output_path,
                    merged[0],
                )
                return output_path

# Define Gradio interface
title = "Depth Any Video with Scalable Synthetic Data"
description = """
Upload a video or image to perform depth estimation using the Depth Any Video model.
Adjust the parameters as needed to control the inference process.
"""
iface = gr.Interface(
    fn=depth_any_video,
    inputs=[
        gr.File(label="Upload Video/Image"),
        gr.Slider(1, 10, step=1, value=3, label="Denoise Steps"),
        gr.Slider(16, 64, step=1, value=32, label="Number of Frames"),
        gr.Slider(8, 32, step=1, value=16, label="Decode Chunk Size"),
        gr.Slider(8, 32, step=1, value=16, label="Number of Interpolation Frames"),
        gr.Slider(2, 10, step=1, value=6, label="Number of Overlap Frames"),
        gr.Slider(512, 2048, step=64, value=1024, label="Maximum Resolution"),
        gr.Number(label="Random Seed (Optional)")
    ],
    outputs=gr.Video(label="Depth Enhanced Video") if DEVICE.type == 'cuda' else gr.Image(label="Depth Enhanced Image"),
    title=title,
    description=description,
    examples=[[os.path.join("assets", f)] for f in os.listdir("assets")],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()

