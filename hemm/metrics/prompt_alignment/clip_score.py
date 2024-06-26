from functools import partial
from typing import Any, Dict, Union

import numpy as np
import torch
import weave
from PIL import Image
from torchmetrics.functional.multimodal import clip_score

from .base import BasePromptAlignmentMetric


class CLIPScoreMetric(BasePromptAlignmentMetric):
    """[CLIP score](https://arxiv.org/abs/2104.08718) metric for text-to-image similarity.
    CLIP Score is a reference free metric that can be used to evaluate the correlation between
    a generated caption for an image and the actual content of the image. It has been found to
    be highly correlated with human judgement.

    Args:
        name (str, optional): Name of the metric. Defaults to "clip_score".
        clip_model_name_or_path (str, optional): The name or path of the CLIP model to use.
            Defaults to "openai/clip-vit-base-patch16".
    """

    def __init__(
        self,
        clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
        name: str = "clip_score",
    ) -> None:
        super().__init__(name)
        self.clip_score_fn = partial(
            clip_score, model_name_or_path=clip_model_name_or_path
        )
        self.config = {"clip_model_name_or_path": clip_model_name_or_path}

    @weave.op()
    def compute_metric(
        self, pil_image: Image.Image, prompt: str
    ) -> Union[float, Dict[str, float]]:
        images = np.expand_dims(np.array(pil_image), axis=0)
        return float(
            self.clip_score_fn(
                torch.from_numpy(images).permute(0, 3, 1, 2), prompt
            ).detach()
        )

    @weave.op()
    def evaluate(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, float]:
        _ = "CLIPScoreMetric"
        return super().evaluate(prompt, model_output)

    @weave.op()
    async def evaluate_async(
        self, prompt: str, model_output: Dict[str, Any]
    ) -> Dict[str, float]:
        _ = "CLIPScoreMetric"
        return self.evaluate(prompt, model_output)
