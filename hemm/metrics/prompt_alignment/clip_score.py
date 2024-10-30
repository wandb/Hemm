from functools import partial
from typing import Any, Callable, Dict, Union

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
        model_name (str, optional): The name or path of the CLIP model to use.
    """

    model_name: str
    _clip_score_fn: Callable

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16") -> None:
        super().__init__(model_name=model_name)
        self._clip_score_fn = partial(clip_score, model_name_or_path=model_name)

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
        return super().evaluate(prompt, model_output)
