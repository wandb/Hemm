import base64
from io import BytesIO
from functools import partial
from PIL import Image
from tqdm.auto import tqdm
from typing import Any, Dict

import numpy as np
import weave

import torch
from torch.nn import functional as F
from torchmetrics.functional.multimodal import clip_score, clip_image_quality_assessment
from transformers import BlipForConditionalGeneration, BlipProcessor


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


class BLIPScorer:
    def __init__(
        self,
        blip_model_name_or_path: str = "Salesforce/blip-image-captioning-base",
        name: str = "blip_score",
        device: str = "cuda",
    ) -> None:
        self.scores = []
        self.name = name
        self.device = device
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name_or_path)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name_or_path
        ).to(self.device)
        self.config = {"blip_model_name_or_path": blip_model_name_or_path}

    def compute_blip_score(self, prompt: str, pil_image: Image.Image):
        pixel_values = self.blip_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        text_input_ids = self.blip_processor(
            text=prompt, return_tensors="pt", padding=True, truncation=True
        ).input_ids
        outputs = self.blip_model(
            pixel_values=pixel_values.to(self.device),
            input_ids=text_input_ids.to(self.device),
        )
        logits = outputs.logits[:, :-1, :]
        shift_labels = text_input_ids[..., 1:].contiguous()
        blip_score = float(
            F.cross_entropy(
                logits.view(-1, logits.size(-1)).to(self.device),
                shift_labels.view(-1).to(self.device),
            )
            .detach()
            .item()
        )
        return blip_score

    @weave.op()
    async def __call__(
        self, prompt: str, model_output: Dict[str, Any] = None
    ) -> Dict[str, float]:
        pil_image = Image.open(
            BytesIO(base64.b64decode(model_output["image"].split(";base64,")[-1]))
        )
        blip_score = self.compute_blip_score(prompt, pil_image)
        print(f"{blip_score=}")
        self.scores.append(blip_score)
        return {self.name: blip_score}
