import io
import os
from typing import Any, Dict

import requests
import weave
from PIL import Image

STABILITY_MODEL_HOST = {
    "sd3-large": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
    "sd3-large-turbo": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
}


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
