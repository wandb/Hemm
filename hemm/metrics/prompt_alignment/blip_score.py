from typing import Any, Dict, Union

import weave
from PIL import Image
from torch.nn import functional as F
from transformers import BlipForConditionalGeneration, BlipProcessor

from .base import BasePromptAlignmentMetric


class BLIPScoreMertric(BasePromptAlignmentMetric):
    def __init__(
        self,
        name: str = "blip_score",
        blip_model_name_or_path: str = "Salesforce/blip-image-captioning-base",
        device: str = "cuda",
    ) -> None:
        super().__init__(name)
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name_or_path)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name_or_path
        ).to(device)
        self.config = {"blip_model_name_or_path": blip_model_name_or_path}

    @weave.op()
    def compute_metric(
        self, pil_image: Image, prompt: str
    ) -> Union[float, Dict[str, Any]]:
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
        return float(
            F.cross_entropy(
                logits.view(-1, logits.size(-1)).to(self.device),
                shift_labels.view(-1).to(self.device),
            )
            .detach()
            .item()
        )

    @weave.op()
    def __call__(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, float]:
        _ = "BLIPScoreMertric"
        return super().__call__(prompt, model_output)
