from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import weave
from PIL import Image
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity

from ...utils import base64_encode_image
from .base import BaseImageQualityMetric, ComputeMetricOutput


class LPIPSMetric(BaseImageQualityMetric):
    """LPIPS Metric to compute the Learned Perceptual Image Patch Similarity (LPIPS) score
    between two images. LPIPS essentially computes the similarity between the activations of
    two image patches for some pre-defined network. This measure has been shown to match
    human perception well. A low LPIPS score means that image patches are perceptual similar.

    Args:
        lpips_net_type (str): The network type to use for computing LPIPS. One of "alex", "vgg",
            or "squeeze".
        image_size (Tuple[int, int]): The size to which images will be resized before computing
            LPIPS.
        name (str): The name of the metric.
    """

    def __init__(
        self,
        lpips_net_type: Literal["alex", "vgg", "squeeze"] = "alex",
        image_size: Optional[Tuple[int, int]] = (512, 512),
        name: str = "alexnet_learned_perceptual_image_patch_similarity",
    ) -> None:
        super().__init__(name)
        self.image_size = image_size
        self.lpips_metric = partial(
            learned_perceptual_image_patch_similarity, net_type=lpips_net_type
        )
        self.config = {"lpips_net_type": lpips_net_type}

    @weave.op()
    def compute_metric(
        self, ground_truth_pil_image: Image, generated_pil_image: Image, prompt: str
    ) -> ComputeMetricOutput:
        ground_truth_image = (
            torch.from_numpy(
                np.expand_dims(
                    np.array(ground_truth_pil_image.resize(self.image_size)), axis=0
                ).astype(np.uint8)
            )
            .permute(0, 3, 2, 1)
            .float()
        )
        generated_image = (
            torch.from_numpy(
                np.expand_dims(
                    np.array(generated_pil_image.resize(self.image_size)), axis=0
                ).astype(np.uint8)
            )
            .permute(0, 3, 2, 1)
            .float()
        )
        ground_truth_image = (ground_truth_image / 127.5) - 1.0
        generated_image = (generated_image / 127.5) - 1.0
        return ComputeMetricOutput(
            score=float(
                self.lpips_metric(generated_image, ground_truth_image).detach()
            ),
            ground_truth_image=base64_encode_image(ground_truth_pil_image),
        )

    @weave.op()
    def __call__(
        self, prompt: str, ground_truth_image: str, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        _ = "LPIPSMetric"
        return super().__call__(prompt, ground_truth_image, model_output)
