from functools import partial
from typing import Any, Callable, Dict, Literal, Union

import numpy as np
import torch
import weave
from PIL import Image
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity

from .base import BaseImageQualityMetric, ComputeMetricOutput


class LPIPSMetric(BaseImageQualityMetric):
    """LPIPS Metric to compute the Learned Perceptual Image Patch Similarity (LPIPS) score
    between two images. LPIPS essentially computes the similarity between the activations of
    two image patches for some pre-defined network. This measure has been shown to match
    human perception well. A low LPIPS score means that image patches are perceptual similar.

    Args:
        lpips_net_type (str): The network type to use for computing LPIPS. One of "alex", "vgg",
            or "squeeze".
        image_height (int): The height to which images will be resized before computing LPIPS.
        image_width (int): The width to which images will be resized before computing LPIPS.
    """

    lpips_net_type: Literal["alex", "vgg", "squeeze"]
    image_height: int
    image_width: int
    _lpips_metric: Callable

    def __init__(
        self,
        lpips_net_type: Literal["alex", "vgg", "squeeze"] = "alex",
        image_height: int = 512,
        image_width: int = 512,
    ) -> None:
        super().__init__(
            lpips_net_type=lpips_net_type,
            image_height=image_height,
            image_width=image_width,
        )
        self._lpips_metric = partial(
            learned_perceptual_image_patch_similarity, net_type=self.lpips_net_type
        )

    @weave.op()
    def compute_metric(
        self, ground_truth_pil_image: Image, generated_pil_image: Image
    ) -> ComputeMetricOutput:
        ground_truth_image = (
            torch.from_numpy(
                np.expand_dims(
                    np.array(
                        ground_truth_pil_image.resize(
                            (self.image_height, self.image_width)
                        )
                    ),
                    axis=0,
                ).astype(np.uint8)
            )
            .permute(0, 3, 2, 1)
            .float()
        )
        generated_image = (
            torch.from_numpy(
                np.expand_dims(
                    np.array(
                        generated_pil_image.resize(
                            (self.image_height, self.image_width)
                        )
                    ),
                    axis=0,
                ).astype(np.uint8)
            )
            .permute(0, 3, 2, 1)
            .float()
        )
        ground_truth_image = (ground_truth_image / 127.5) - 1.0
        generated_image = (generated_image / 127.5) - 1.0
        return {
            "score": float(
                self._lpips_metric(generated_image, ground_truth_image).detach()
            ),
            "ground_truth_image": ground_truth_pil_image,
        }

    @weave.op()
    def evaluate(
        self, prompt: str, ground_truth_image: Image.Image, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        return super().evaluate(prompt, ground_truth_image, model_output)
