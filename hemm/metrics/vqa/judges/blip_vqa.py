from typing import Dict, List

import torch
import torch.nn.functional as F
import weave
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor


class BlipVQAJudge(weave.Model):
    """Weave Model to judge the presence of entities in an image using the
    [Blip-VQA model](https://huggingface.co/Salesforce/blip-vqa-base).

    Args:
        blip_processor_address (str): The address of the BlipProcessor model.
        blip_vqa_address (str): The address of the BlipForQuestionAnswering model.
        device (str): The device to use for inference
    """

    blip_processor_address: str
    blip_vqa_address: str
    device: str
    _torch_dtype: torch.dtype = torch.float32
    _blip_processor_model: BlipProcessor = None
    _blip_vqa_model: BlipForQuestionAnswering = None

    def __init__(
        self,
        blip_processor_address: str = "Salesforce/blip-vqa-base",
        blip_vqa_address: str = "Salesforce/blip-vqa-base",
        device: str = "cuda",
    ):
        super().__init__(
            blip_processor_address=blip_processor_address,
            blip_vqa_address=blip_vqa_address,
            device=device,
        )
        self._blip_processor_model = BlipProcessor.from_pretrained(
            self.blip_processor_address
        )
        self._blip_vqa_model = BlipForQuestionAnswering.from_pretrained(
            self.blip_vqa_address, torch_dtype=self._torch_dtype
        ).to(self.device)

    def _get_probability(self, target_token: str, scores: List[torch.Tensor]) -> float:
        target_token_id = self._blip_processor_model.tokenizer.convert_tokens_to_ids(
            target_token
        )
        probabilities = [F.softmax(score, dim=-1) for score in scores]
        target_token_probabilities = [
            prob[:, target_token_id].item() for prob in probabilities
        ]
        max_target_token_probability = max(target_token_probabilities)
        return max_target_token_probability

    @weave.op()
    def get_target_token_probability(
        self, question: str, image: Image.Image
    ) -> Dict[str, float]:
        inputs = self._blip_processor_model(image, question, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generated_ids = self._blip_vqa_model.generate(
                **inputs, output_scores=True, return_dict_in_generate=True
            )
        scores = generated_ids.scores
        yes_probability = self._get_probability("yes", scores)
        no_probability = self._get_probability("no", scores)
        return {
            "yes_proba": yes_probability,
            "no_proba": no_probability,
            "present": yes_probability > no_probability,
        }

    @weave.op()
    def predict(
        self, adj_1: str, noun_1: str, adj_2: str, noun_2: str, image: Image.Image
    ) -> Dict:
        """Predict the probabilities presence of entities in an image using the Blip-VQA model.

        Args:
            adj_1 (str): The adjective of the first entity.
            noun_1 (str): The noun of the first entity.
            adj_2 (str): The adjective of the second entity.
            noun_2 (str): The noun of the second entity.
            image (Image.Image): The input image.

        Returns:
            Dict: The probabilities of the presence of the entities.
        """
        question_1 = f"is {adj_1} {noun_1} present in the picture?"
        question_2 = f"is {adj_2} {noun_2} present in the picture?"
        return {
            "entity_1": self.get_target_token_probability(question_1, image),
            "entity_2": self.get_target_token_probability(question_2, image),
        }
