from abc import ABC, abstractmethod
import base64
from io import BytesIO
from typing import Any, Dict, Union

import weave
from PIL import Image


class BaseImageQualityMetric(ABC):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.scores = []
        self.name = name
        self.config = {}

    @abstractmethod
    def compute_metric(
        self,
        ground_truth_pil_image: Image.Image,
        generated_pil_image: Image.Image,
        prompt: str,
    ) -> Union[float, Dict[str, float]]:
        pass

    @weave.op()
    async def __call__(
        self, prompt: str, ground_truth_image: str, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        ground_truth_pil_image = Image.open(
            BytesIO(base64.b64decode(ground_truth_image.split(";base64,")[-1]))
        )
        generated_pil_image = Image.open(
            BytesIO(base64.b64decode(model_output["image"].split(";base64,")[-1]))
        )
        score = self.compute_metric(ground_truth_pil_image, generated_pil_image, prompt)
        self.scores.append(score)
        return {self.name: score}
