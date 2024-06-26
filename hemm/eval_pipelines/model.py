from typing import Dict

import torch
import weave
from diffusers import DiffusionPipeline

from ..utils import base64_encode_image


class BaseDiffusionModel(weave.Model):
    """Base `weave.Model` wrapping `diffusers.DiffusionPipeline`.

    Args:
        diffusion_model_name_or_path (str): The name or path of the diffusion model.
        enable_cpu_offfload (bool): Enable CPU offload for the diffusion model.
        image_height (int): The height of the generated image.
        image_width (int): The width of the generated image.
        disable_safety_checker (bool): Disable safety checker for the diffusion model.
    """

    diffusion_model_name_or_path: str
    enable_cpu_offfload: bool = False
    image_height: int = 512
    image_width: int = 512
    disable_safety_checker: bool = True
    _torch_dtype: torch.dtype = torch.float16
    _pipeline: DiffusionPipeline = None

    def initialize(self) -> None:
        pipeline_init_kwargs = {
            "pretrained_model_name_or_path": self.diffusion_model_name_or_path,
            "torch_dtype": self._torch_dtype,
        }
        if self.disable_safety_checker:
            pipeline_init_kwargs["safety_checker"] = None
        self._pipeline = DiffusionPipeline.from_pretrained(**pipeline_init_kwargs)
        if self.enable_cpu_offfload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline = self._pipeline.to("cuda")
        self._pipeline.set_progress_bar_config(leave=False, desc="Generating Image")

    @weave.op()
    def predict(self, prompt: str, seed: int) -> Dict[str, str]:
        pipeline_output = self._pipeline(
            prompt,
            num_images_per_prompt=1,
            height=self.image_height,
            width=self.image_width,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        return {
            "image": base64_encode_image(pipeline_output.images[0]),
            "nsfw_content_detected": pipeline_output.nsfw_content_detected is not None,
        }
