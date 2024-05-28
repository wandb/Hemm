from abc import ABC, abstractmethod
import base64
from io import BytesIO
from PIL import Image
from typing import Any, Dict, Union

import weave


class BasePromptAlignmentMetric(ABC):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.scores = []
        self.name = name
        self.config = {}

    @abstractmethod
    def compute_metric(
        self, pil_image: Image.Image, prompt: str
    ) -> Union[float, Dict[str, float]]:
        pass

    @weave.op()
    async def __call__(
        self, prompt: str, model_output: Dict[str, Any]
    ) -> Dict[str, float]:
        pil_image = Image.open(
            BytesIO(base64.b64decode(model_output["image"][0].split(";base64,")[-1]))
        )
        score = self.compute_metric(pil_image, prompt)
        self.scores.append(score)
        return {self.name: score}
