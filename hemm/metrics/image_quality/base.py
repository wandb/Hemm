from abc import ABC, abstractmethod
import base64
from io import BytesIO
from typing import Any, Dict, Union

import weave
from PIL import Image


class BaseImageQualityMetric(ABC):

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
    ) -> Union[float, Dict[str, float]]:
        """Compute the metric for the given images. This is an abstract
        method and must be overriden by the child class implementation.
        
        Args:
            ground_truth_pil_image (Image.Image): Ground truth image in PIL format.
            generated_pil_image (Image.Image): Generated image in PIL format.
            prompt (str): Prompt for the image generation.
        
        Returns:
            Union[float, Dict[str, float]]: Metric score.
        """
        pass

    @weave.op()
    async def __call__(
        self, prompt: str, ground_truth_image: str, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
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
        score = self.compute_metric(ground_truth_pil_image, generated_pil_image, prompt)
        self.scores.append(score)
        return {self.name: score}
