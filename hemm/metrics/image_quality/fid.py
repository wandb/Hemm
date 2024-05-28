from typing import Any, Dict, Tuple, Union
from PIL import Image

import numpy as np
import torch
from torchmetrics.image import FrechetInceptionDistance

import weave

from .base import BaseImageQualityMetric, ComputeMetricOutput
from ...utils import base64_encode_image


class FIDScore(BaseImageQualityMetric):

    def __init__(
        self,
        feature: int = 2048,
        reset_real_features: bool = True,
        image_size: Tuple[int, int] = (256, 256),
        name: str = "fid_score",
    ) -> None:
        super().__init__(name)
        self.image_size = image_size
        self.stateful_fid_metric = FrechetInceptionDistance(
            feature=feature, reset_real_features=reset_real_features, normalize=True
        )
        self.config = {
            "fid_feature": feature,
            "fid_reset_real_features": reset_real_features,
            "fid_image_size": image_size,
        }

    @weave.op()
    def compute_metric(
        self,
        ground_truth_pil_image: Image.Image,
        generated_pil_image: Image.Image,
        prompt: str,
    ) -> ComputeMetricOutput:
        ground_truth_image = (
            torch.from_numpy(
                np.expand_dims(
                    np.array(ground_truth_pil_image.resize(self.image_size)), axis=0
                ).astype(np.uint8)
            )
            .permute(0, 3, 2, 1)
            .float()
        ) / 255.0
        generated_image = (
            torch.from_numpy(
                np.expand_dims(
                    np.array(generated_pil_image.resize(self.image_size)), axis=0
                ).astype(np.uint8)
            )
            .permute(0, 3, 2, 1)
            .float()
        ) / 255.0
        ground_truth_image = torch.concatenate([ground_truth_image, ground_truth_image])
        generated_image = torch.concatenate([generated_image, generated_image])
        self.stateful_fid_metric.update(ground_truth_image, real=True)
        self.stateful_fid_metric.update(generated_image, real=False)
        return ComputeMetricOutput(
            score=float(self.stateful_fid_metric.compute().detach()),
            ground_truth_image=base64_encode_image(ground_truth_pil_image),
        )

    @weave.op()
    async def __call__(
        self, prompt: str, ground_truth_image: str, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        _ = "FIDScore"
        return super().__call__(prompt, ground_truth_image, model_output)
