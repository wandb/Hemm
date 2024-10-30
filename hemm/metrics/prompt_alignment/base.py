from abc import abstractmethod
from typing import Any, Dict, Union

import weave
from PIL import Image


class BasePromptAlignmentMetric(weave.Scorer):

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

    def evaluate(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, float]:
        """Compute the metric for the given image. This method is used as the scorer
        function for `weave.Evaluation` in the evaluation pipelines.

        Args:
            prompt (str): Prompt for the image generation.
            model_output (Dict[str, Any]): Model output containing the generated image.

        Returns:
            Dict[str, float]: Metric score.
        """
        score = self.compute_metric(model_output["image"], prompt)
        return {self.name: score}
