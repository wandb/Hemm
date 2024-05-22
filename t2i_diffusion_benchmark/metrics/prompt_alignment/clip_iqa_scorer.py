import base64
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Any, Dict

import numpy as np
import torch
import weave
from tqdm.auto import tqdm
from torchmetrics.functional.multimodal import clip_image_quality_assessment


class CLIPImageQualityScorer:

    def __init__(
        self,
        clip_model_name_or_path: str = "clip_iqa",
        name: str = "clip_image_quality_assessment",
    ) -> None:
        self.scores = []
        self.prompt = "quality"
        self.name = name
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

    @weave.op()
    async def __call__(self, model_output: Dict[str, Any]) -> Dict[str, float]:
        pil_image = Image.open(
            BytesIO(base64.b64decode(model_output["image"].split(";base64,")[-1]))
        )
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
        self.scores.append(score_dict)
        return score_dict
