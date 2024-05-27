from functools import partial
from PIL import Image
from typing import Dict, Union

import numpy as np
import torch
from torchmetrics.functional.multimodal import clip_score

from .base import BasePromptAlignmentMetric


class CLIPScoreMetric(BasePromptAlignmentMetric):

    def __init__(
        self,
        name: str = "clip_score",
        clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
    ) -> None:
        super().__init__(name)
        self.clip_score_fn = partial(
            clip_score, model_name_or_path=clip_model_name_or_path
        )
        self.config = {"clip_model_name_or_path": clip_model_name_or_path}

    def compute_metric(
        self, pil_image: Image.Image, prompt: str
    ) -> Union[float, Dict[str, float]]:
        images = np.expand_dims(np.array(pil_image), axis=0)
        return float(
            self.clip_score_fn(
                torch.from_numpy(images).permute(0, 3, 1, 2), prompt
            ).detach()
        )
