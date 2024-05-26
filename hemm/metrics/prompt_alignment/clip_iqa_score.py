import base64
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Any, Dict, Union

import numpy as np
import torch
import weave
from tqdm.auto import tqdm
from torchmetrics.functional.multimodal import clip_image_quality_assessment

from .base import BasePromptAlignmentMetric


class CLIPImageQualityScoreMetric(BasePromptAlignmentMetric):
    def __init__(
        self,
        name: str = "clip_image_quality_assessment",
        clip_model_name_or_path: str = "clip_iqa",
    ) -> None:
        super().__init__(name)
        self.clip_iqa_fn = partial(
            clip_image_quality_assessment, model_name_or_path=clip_model_name_or_path
        )
        self.built_in_prompts = [
            "quality",
            "brightness",
            "noisiness",
            "colorfullness",
            "sharpness",
            "contrast",
            "complexity",
            "natural",
            "happy",
            "scary",
            "new",
            "real",
            "beautiful",
            "lonely",
            "relaxing",
        ]
        self.config = {"clip_model_name_or_path": clip_model_name_or_path}

    def compute_metric(
        self, pil_image: Image, prompt: str
    ) -> Union[float, Dict[str, float]]:
        images = np.expand_dims(np.array(pil_image), axis=0).astype(np.uint8) / 255.0
        score_dict = {}
        for prompt in tqdm(
            self.built_in_prompts, desc="Calculating IQA scores", leave=False
        ):
            clip_iqa_score = float(
                self.clip_iqa_fn(
                    images=torch.from_numpy(images).permute(0, 3, 1, 2),
                    prompts=tuple([self.prompt] * images.shape[0]),
                ).detach()
            )
            score_dict[f"{self.name}_{prompt}"] = clip_iqa_score
        return score_dict
