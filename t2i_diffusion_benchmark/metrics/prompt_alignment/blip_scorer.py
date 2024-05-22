import base64
from io import BytesIO
from PIL import Image
from typing import Any, Dict

import weave
from torch.nn import functional as F
from transformers import BlipForConditionalGeneration, BlipProcessor


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
