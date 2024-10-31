from typing import Any, Dict

import weave
from torch.nn import functional as F
from transformers import BlipForConditionalGeneration, BlipProcessor


class BLIPScoreMertric(weave.Scorer):
    model_name: str = "Salesforce/blip-image-captioning-base"
    device: str = "cuda"
    _blip_processor: BlipProcessor
    _blip_model: BlipForConditionalGeneration

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cuda",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self._blip_processor = BlipProcessor.from_pretrained(model_name)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            device
        )

    @weave.op()
    def score(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, float]:
        pixel_values = self.blip_processor(
            images=model_output["image"], return_tensors="pt"
        ).pixel_values
        text_input_ids = self._blip_processor(
            text=prompt, return_tensors="pt", padding=True, truncation=True
        ).input_ids
        outputs = self._blip_model(
            pixel_values=pixel_values.to(self.device),
            input_ids=text_input_ids.to(self.device),
        )
        logits = outputs.logits[:, :-1, :]
        shift_labels = text_input_ids[..., 1:].contiguous()
        return {
            "score": float(
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)).to(self.device),
                    shift_labels.view(-1).to(self.device),
                )
                .detach()
                .item()
            )
        }
