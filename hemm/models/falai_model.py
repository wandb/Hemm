from typing import Any, Dict

import fal_client
import weave
from diffusers.utils.loading_utils import load_image
from PIL import Image

from ..utils import custom_weave_wrapper


class FalAIModel(weave.Model):
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
