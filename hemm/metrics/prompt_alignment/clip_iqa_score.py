from functools import partial
from PIL import Image
from typing import Dict, Union

import numpy as np
import torch
from tqdm.auto import tqdm
from torchmetrics.functional.multimodal import clip_image_quality_assessment

from .base import BasePromptAlignmentMetric


class CLIPImageQualityScoreMetric(BasePromptAlignmentMetric):
    """[CLIP Image Quality Assessment](https://arxiv.org/abs/2207.12396) metric
    for to measuring the visual content of images.

    The metric is based on the [CLIP](https://arxiv.org/abs/2103.00020) model,
    which is a neural network trained on a variety of (image, text) pairs to be
    able to generate a vector representation of the image and the text that is
    similar if the image and text are semantically similar.

    The metric works by calculating the cosine similarity between user provided images
    and pre-defined prompts. The prompts always comes in pairs of “positive” and “negative”
    such as “Good photo.” and “Bad photo.”. By calculating the similartity between image
    embeddings and both the “positive” and “negative” prompt, the metric can determine which
    prompt the image is more similar to. The metric then returns the probability that the
    image is more similar to the first prompt than the second prompt.

    Args:
        clip_model_name_or_path (str, optional): The name or path of the CLIP model to use.
            Defaults to "clip_iqa".
        name (str, optional): Name of the metric. Defaults to "clip_image_quality_assessment".
    """

    def __init__(
        self,
        clip_model_name_or_path: str = "clip_iqa",
        name: str = "clip_image_quality_assessment",
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
                    prompts=tuple([prompt] * images.shape[0]),
                ).detach()
            )
            score_dict[f"{self.name}_{prompt}"] = clip_iqa_score
        return score_dict
