import io
import os
from typing import Any, Dict

import fal_client
import requests
import torch
import weave
from diffusers import DiffusionPipeline
from diffusers.utils.loading_utils import load_image
from PIL import Image

from ..utils import custom_weave_wrapper


STABILITY_MODEL_HOST = {
    "sd3-large": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
    "sd3-large-turbo": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
}


class BaseDiffusionModel(weave.Model):
    """`weave.Model` wrapping `diffusers.DiffusionPipeline`.

    Args:
        diffusion_model_name_or_path (str): The name or path of the diffusion model.
        enable_cpu_offfload (bool): Enable CPU offload for the diffusion model.
        image_height (int): The height of the generated image.
        image_width (int): The width of the generated image.
        num_inference_steps (int): The number of inference steps.
        disable_safety_checker (bool): Disable safety checker for the diffusion model.
        configs (Dict[str, Any]): Additional configs.
        pipeline_configs (Dict[str, Any]): Diffusion pipeline configs.
        inference_kwargs (Dict[str, Any]): Inference kwargs.
    """

    diffusion_model_name_or_path: str
    enable_cpu_offfload: bool = False
    image_height: int = 512
    image_width: int = 512
    num_inference_steps: int = 50
    disable_safety_checker: bool = True
    configs: Dict[str, Any] = {}
    pipeline_configs: Dict[str, Any] = {}
    inference_kwargs: Dict[str, Any] = {}
    _torch_dtype: torch.dtype = torch.float16
    _pipeline: DiffusionPipeline = None

    def __init__(
        self,
        diffusion_model_name_or_path: str,
        enable_cpu_offfload: bool = False,
        image_height: int = 512,
        image_width: int = 512,
        num_inference_steps: int = 50,
        disable_safety_checker: bool = True,
        configs: Dict[str, Any] = {},
        pipeline_configs: Dict[str, Any] = {},
        inference_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            diffusion_model_name_or_path=diffusion_model_name_or_path,
            enable_cpu_offfload=enable_cpu_offfload,
            image_height=image_height,
            image_width=image_width,
            num_inference_steps=num_inference_steps,
            disable_safety_checker=disable_safety_checker,
            configs=configs,
            pipeline_configs=pipeline_configs,
            inference_kwargs=inference_kwargs,
        )
        self.configs["torch_dtype"] = str(self._torch_dtype)
        pipeline_init_kwargs = {
            "pretrained_model_name_or_path": self.diffusion_model_name_or_path,
            "torch_dtype": self._torch_dtype,
        }
        pipeline_init_kwargs.update(self.pipeline_configs)
        if self.disable_safety_checker:
            pipeline_init_kwargs["safety_checker"] = None
        self._pipeline = DiffusionPipeline.from_pretrained(**pipeline_init_kwargs)
        if self.enable_cpu_offfload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline = self._pipeline.to("cuda")
        self._pipeline.set_progress_bar_config(leave=False, desc="Generating Image")

    @weave.op()
    def predict(self, prompt: str, seed: int) -> Dict[str, Any]:
        pipeline_output = self._pipeline(
            prompt,
            num_images_per_prompt=1,
            height=self.image_height,
            width=self.image_width,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            num_inference_steps=self.num_inference_steps,
            **self.inference_kwargs,
        )
        return {"image": pipeline_output.images[0]}


class FalDiffusionModel(weave.Model):
    """`weave.Model` wrapping [FalAI](https://fal.ai/) calls.

    Args:
        model_name (str): FalAI model name.
        inference_kwargs (Dict[str, Any]): Inference kwargs.
    """

    model_name: str
    inference_kwargs: Dict[str, Any] = {}

    @weave.op()
    def generate_image(self, prompt: str, seed: int) -> Image.Image:
        result = custom_weave_wrapper(name="fal_client.submit.get")(
            fal_client.submit(
                self.model_name,
                arguments={"prompt": prompt, "seed": seed, **self.inference_kwargs},
            ).get
        )()
        return load_image(result["images"][0]["url"])

    @weave.op()
    def predict(self, prompt: str, seed: int) -> Image.Image:
        return {"image": self.generate_image(prompt=prompt, seed=seed)}


class StabilityAPIModel(weave.Model):
    """`weave.Model` wrapping Stability API calls.

    Args:
        model_name (str): Stability model name.
        aspect_ratio (str): Aspect ratio of the generated image.
        creativity (float): Creativity of the generated image.
    """

    model_name: str
    aspect_ratio: str = "1:1"
    creativity: float = 0.35
    configs: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str,
        aspect_ratio: str = "1:1",
        creativity: float = 0.35,
    ) -> None:
        assert aspect_ratio in [
            "1:1",
            "16:9",
            "21:9",
            "2:3",
            "3:2",
            "4:5",
            "5:4",
            "9:16",
            "9:21",
        ], "Invalid aspect ratio"
        super().__init__(
            model_name=model_name, aspect_ratio=aspect_ratio, creativity=creativity
        )

    @weave.op()
    def send_generation_request(self, prompt: str, seed: int):
        api_key = os.environ["STABILITY_KEY"]
        headers = {"Accept": "image/*", "Authorization": f"Bearer {api_key}"}
        response = requests.post(
            STABILITY_MODEL_HOST[self.model_name],
            headers=headers,
            files={"none": ""},
            data={
                "prompt": prompt,
                "negative_prompt": "",
                "aspect_ratio": self.aspect_ratio,
                "seed": seed,
                "output_format": "png",
                "model": self.model_name,
                "mode": "text-to-image",
                "creativity": self.creativity,
            },
        )
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        return response

    @weave.op()
    def predict(self, prompt: str, seed: int) -> Image.Image:
        response = self.send_generation_request(prompt=prompt, seed=seed)
        image = Image.open(io.BytesIO(response.content))
        return {"image": image}
