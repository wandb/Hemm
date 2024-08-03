import base64
import io
import os
from pathlib import Path
from typing import Optional, Union

import fire
import instructor
import torch
import weave
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel

EXT_TO_MIMETYPE = {
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def base64_encode_image(
    image_path: Union[str, Image.Image], mimetype: Optional[str] = None
) -> str:
    image = Image.open(image_path) if isinstance(image_path, str) else image_path
    mimetype = (
        EXT_TO_MIMETYPE[Path(image_path).suffix]
        if isinstance(image_path, str)
        else "image/png"
    )
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    encoded_string = f"data:{mimetype};base64,{encoded_string}"
    return str(encoded_string)


class SD3Model(weave.Model):
    model_name_or_path: str
    enable_cpu_offfload: bool
    _pipeline: StableDiffusion3Pipeline

    def __init__(self, model_name_or_path: str, enable_cpu_offfload: bool):
        super().__init__(
            model_name_or_path=model_name_or_path,
            enable_cpu_offfload=enable_cpu_offfload,
        )
        self._pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.float16
        )
        if self.enable_cpu_offfload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline = self._pipeline.to("cuda")

    @weave.op()
    def predict(
        self,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        image_size: int,
        guidance_scale: float,
    ) -> str:
        image = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
        ).images[0]
        return base64_encode_image(image)


class JudgeMent(BaseModel):
    score: float
    explanation: str


class OpenAIModel(weave.Model):
    model: str
    max_retries: int = 5
    seed: int = 42
    _openai_client: instructor.Instructor

    def __init__(self, model: str):
        super().__init__(model=model)
        self._openai_client = instructor.from_openai(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        )

    @weave.op()
    def predict(self, prompt: str, image: str) -> JudgeMent:
        return self._openai_client.chat.completions.create(
            model=self.model,
            response_model=JudgeMent,
            max_retries=self.max_retries,
            seed=self.seed,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful assistant meant to describe images is detail. You should pay special attention to the objects
in the image and their attributes (such as color, shape, texture), spatial layout and action relationships.
You have to extract the the score and the explanation from the response.
                    """,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
How well the image align with the prompt '{prompt}'. Assign a score between 0 and 1.
Also provide a detailed explanation to justify your score.
                            """,
                        },
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                },
            ],
        )


weave.init(project_name="intro-example")

diffusion_model = SD3Model(
    model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
    enable_cpu_offfload=True,
)
judge_model = OpenAIModel(model="gpt-4-turbo")


@weave.op()
def evaluate(prompt: str) -> JudgeMent:
    image = diffusion_model.predict(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        image_size=1024,
        guidance_scale=7.0,
    )
    return judge_model.predict(prompt=prompt, image=image)


fire.Fire(evaluate)
