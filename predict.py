# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from typing import List
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler

MODEL_NAME = "segmind/SSD-1B"
MODEL_CACHE = "model-cache"
MODEL_UNET = "latent-consistency/lcm-ssd-1b"
UNET_CACHE = "unet-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_UNET,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=UNET_CACHE
        )
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=MODEL_CACHE
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a close-up picture of an old man standing in the rain"
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1, le=10, default=4,
        ),
        guidance_scale: float = Input(
            description="Factor to scale image by", 
            ge=0, le=10, default=8.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = self.pipe(**common_args)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
        
