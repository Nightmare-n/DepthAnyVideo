# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import numpy as np
import torch
from cog import BasePredictor, Input, Path

from dav.pipelines import DAVPipeline
from dav.models import UNetSpatioTemporalRopeConditionModel
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from dav.utils import img_utils


MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/Nightmare-n/DepthAnyVideo/{MODEL_CACHE}.tar"

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.device_type = "cuda"
        self.device = torch.device(self.device_type)

        model_base = f"{MODEL_CACHE}/hhyangcs/depth-any-video"
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
        self.pipe = DAVPipeline(
            vae=vae,
            unet=unet,
            unet_interp=unet_interp,
            scheduler=scheduler,
        ).to(self.device)

    def predict(
        self,
        input_image_or_video: Path = Input(description="Input image or video"),
        input_is_video: bool = Input(
            description="Specify if the input is a video.", default=True
        ),
        denoise_steps: int = Input(
            description="Number of denoising steps, 1-3 steps work fine", default=3
        ),
        num_frames: int = Input(
            description="Number of frames to infer per forward. This should be an even number",
            default=32,
        ),
        decode_chunk_size: int = Input(
            description="Number of frames to decode per forward", default=16
        ),
        num_interp_frames: int = Input(
            description="Number of frames for inpaint inference", default=16
        ),
        num_overlap_frames: int = Input(
            description="Number of frames to overlap between windows", default=6
        ),
        max_resolution: int = Input(
            description="Maximum resolution for inference", default=1024
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if input_is_video:
            assert num_frames % 2 == 0, "num_frames should be even."
            assert (
                2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
            ), "Invalid frame overlap."
            max_frames = (num_interp_frames + 2 - num_overlap_frames) * (
                num_frames // 2
            )
            image, fps = img_utils.read_video(
                str(input_image_or_video), max_frames=max_frames
            )
        else:
            image = img_utils.read_image(str(input_image_or_video))

        image = img_utils.imresize_max(image, max_resolution)
        image = img_utils.imcrop_multi(image)
        image_tensor = np.ascontiguousarray(
            [_img.transpose(2, 0, 1) / 255.0 for _img in image]
        )
        image_tensor = torch.from_numpy(image_tensor).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device_type, dtype=torch.float16
        ):
            pipe_out = self.pipe(
                image_tensor,
                num_frames=num_frames,
                num_overlap_frames=num_overlap_frames,
                num_interp_frames=num_interp_frames,
                decode_chunk_size=decode_chunk_size,
                num_inference_steps=denoise_steps,
            )

        disparity_colored = pipe_out.disparity_colored
        image = pipe_out.image
        out_path = "/tmp/out.mp4" if input_is_video else "/tmp/out.png"

        if input_is_video:
            img_utils.write_video(out_path, disparity_colored, fps)
        else:
            img_utils.write_image(out_path, disparity_colored[0])
        return Path(out_path)
