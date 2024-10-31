from functools import partial
from typing import Any, Callable, Dict

import numpy as np
import torch
import weave
from torchmetrics.functional.multimodal import clip_score


class CLIPScoreMetric(weave.Scorer):
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
    def score(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, float]:
        images = np.expand_dims(np.array(model_output["image"]), axis=0)
        return {
            "score": float(
                self._clip_score_fn(
                    torch.from_numpy(images).permute(0, 3, 1, 2), prompt
                ).detach()
            )
        }
