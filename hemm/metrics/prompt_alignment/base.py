import base64
from abc import abstractmethod
from io import BytesIO
from typing import Any, Dict, Union

import weave
from PIL import Image

from ..base import BaseMetric


class BasePromptAlignmentMetric(BaseMetric):
    """Base class for Prompt Alignment Metrics.

    Args:
        name (str): Name of the metric.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.scores = []
        self.name = name
        self.config = {}

    @abstractmethod
    def compute_metric(
        self, pil_image: Image.Image, prompt: str
    ) -> Union[float, Dict[str, float]]:
        """Compute the metric for the given image. This is an abstract
        method and must be overriden by the child class implementation.

        Args:
            pil_image (Image.Image): Image in PIL format.
            prompt (str): Prompt for the image generation.

        Returns:
            Union[float, Dict[str, float]]: Metric score.
        """
        pass

    def evaluate(
        self, prompt: str, model_output: Dict[str, Any], metadata: weave.Model
    ) -> Dict[str, float]:
        """Compute the metric for the given image. This method is used as the scorer
        function for `weave.Evaluation` in the evaluation pipelines.

        Args:
            prompt (str): Prompt for the image generation.
            model_output (Dict[str, Any]): Model output containing the generated image.

        Returns:
            Dict[str, float]: Metric score.
        """
        pil_image = Image.open(
            BytesIO(base64.b64decode(model_output["image"].split(";base64,")[-1]))
        )
        score = self.compute_metric(pil_image, prompt)
        self.scores.append(score)
        return {self.name: score}
