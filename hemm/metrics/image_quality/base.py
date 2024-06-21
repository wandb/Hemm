import base64
from abc import abstractmethod
from io import BytesIO
from typing import Any, Dict, Union

from PIL import Image
from pydantic import BaseModel

from ..base import BaseMetric


class ComputeMetricOutput(BaseModel):
    """Output of the metric computation function."""

    score: Union[float, Dict[str, float]]
    ground_truth_image: str


class BaseImageQualityMetric(BaseMetric):

    def __init__(self, name: str) -> None:
        """Base class for Image Quality Metrics.

        Args:
            name (str): Name of the metric.
        """
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
    ) -> ComputeMetricOutput:
        """Compute the metric for the given images. This is an abstract
        method and must be overriden by the child class implementation.

        Args:
            ground_truth_pil_image (Image.Image): Ground truth image in PIL format.
            generated_pil_image (Image.Image): Generated image in PIL format.
            prompt (str): Prompt for the image generation.

        Returns:
            ComputeMetricOutput: Output containing the metric score and ground truth image.
        """
        pass

    def evaluate(
        self, prompt: str, ground_truth_image: str, model_output: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute the metric for the given images. This method is used as the scorer
        function for `weave.Evaluation` in the evaluation pipelines.

        Args:
            prompt (str): Prompt for the image generation.
            ground_truth_image (str): Ground truth image in base64 format.
            model_output (Dict[str, Any]): Model output containing the generated image.

        Returns:
            Union[float, Dict[str, float]]: Metric score.
        """
        ground_truth_pil_image = Image.open(
            BytesIO(base64.b64decode(ground_truth_image.split(";base64,")[-1]))
        )
        generated_pil_image = Image.open(
            BytesIO(base64.b64decode(model_output["image"].split(";base64,")[-1]))
        )
        metric_output = self.compute_metric(
            ground_truth_pil_image, generated_pil_image, prompt
        )
        self.scores.append(metric_output.score)
        return {self.name: metric_output.score}
