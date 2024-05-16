import base64
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Any, Dict

import numpy as np
import weave

import torch
from torchmetrics.functional.multimodal import clip_score


class CLIPScorer:

    def __init__(
        self,
        clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
        name: str = "clip_score",
    ) -> None:
        self.scores = []
        self.name = name
        self.clip_score_fn = partial(
            clip_score, model_name_or_path=clip_model_name_or_path
        )
        self.config = {"clip_model_name_or_path": clip_model_name_or_path}

    @weave.op()
    async def __call__(
        self, prompt: str, model_output: Dict[str, Any]
    ) -> Dict[str, float]:
        pil_image = Image.open(
            BytesIO(base64.b64decode(model_output["image"].split(";base64,")[-1]))
        )
        images = np.expand_dims(np.array(pil_image), axis=0)
        clip_score = float(
            self.clip_score_fn(
                torch.from_numpy(images).permute(0, 3, 1, 2), prompt
            ).detach()
        )
        self.scores.append(clip_score)
        return {self.name: clip_score}
