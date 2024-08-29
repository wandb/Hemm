from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import weave
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure

from ...utils import base64_encode_image
from .base import BaseImageQualityMetric, ComputeMetricOutput


class SSIMMetric(BaseImageQualityMetric):
    """SSIM Metric to compute the
    [Structural Similarity Index Measure (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity)
    between two images.

    Args:
        ssim_gaussian_kernel (bool): Whether to use a Gaussian kernel for SSIM computation.
        ssim_sigma (float): The standard deviation of the Gaussian kernel.
        ssim_kernel_size (int): The size of the Gaussian kernel.
        ssim_data_range (Optional[Union[float, Tuple[float, float]]]): The data range of the input
            image (min, max). If None, the data range is determined from the image data type.
        ssim_k1 (float): The constant used to stabilize the SSIM numerator.
        ssim_k2 (float): The constant used to stabilize the SSIM denominator.
        image_size (Tuple[int, int]): The size to which images will be resized before computing
            SSIM.
        name (str): The name of the metric.
    """

    def __init__(
        self,
        ssim_gaussian_kernel: bool = True,
        ssim_sigma: float = 1.5,
        ssim_kernel_size: int = 11,
        ssim_data_range: Union[float, Tuple[float, float], None] = None,
        ssim_k1: float = 0.01,
        ssim_k2: float = 0.03,
        image_size: Optional[Tuple[int, int]] = (512, 512),
        name: str = "structural_similarity_index_measure",
    ) -> None:
        super().__init__(name)
        self.image_size = image_size
        self.ssim_metric = partial(
            structural_similarity_index_measure,
            gaussian_kernel=ssim_gaussian_kernel,
            sigma=ssim_sigma,
            kernel_size=ssim_kernel_size,
            data_range=ssim_data_range,
            k1=ssim_k1,
            k2=ssim_k2,
        )
        self.config = {
            "ssim_gaussian_kernel": ssim_gaussian_kernel,
            "ssim_sigma": ssim_sigma,
            "ssim_kernel_size": ssim_kernel_size,
            "ssim_data_range": ssim_data_range,
            "ssim_k1": ssim_k1,
            "ssim_k2": ssim_k2,
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
            .permute(0, 3, 1, 2)
            .float()
        )
        generated_image = (
            torch.from_numpy(
                np.expand_dims(
                    np.array(generated_pil_image.resize(self.image_size)), axis=0
                ).astype(np.uint8)
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        return ComputeMetricOutput(
            score=float(self.ssim_metric(generated_image, ground_truth_image)),
            ground_truth_image=base64_encode_image(ground_truth_pil_image),
        )

    @weave.op()
    def evaluate(
        self, prompt: str, ground_truth_image: Image.Image, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        _ = "SSIMMetric"
        return super().evaluate(prompt, ground_truth_image, model_output)

    @weave.op()
    async def evaluate_async(
        self, prompt: str, ground_truth_image: Image.Image, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        _ = "SSIMMetric"
        return self.evaluate(prompt, ground_truth_image, model_output)
