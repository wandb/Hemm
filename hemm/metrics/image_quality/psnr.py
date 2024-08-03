from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import weave
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio

from ...utils import base64_encode_image
from .base import BaseImageQualityMetric, ComputeMetricOutput


class PSNRMetric(BaseImageQualityMetric):
    """PSNR Metric to compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        psnr_data_range (Optional[Union[float, Tuple[float, float]]]): The data range of the input
            image (min, max). If None, the data range is determined from the image data type.
        psnr_base (float): The base of the logarithm in the PSNR formula.
        image_size (Tuple[int, int]): The size to which images will be resized before computing
            PSNR.
        name (str): The name of the metric.
    """

    def __init__(
        self,
        psnr_data_range: Optional[Union[float, Tuple[float, float]]] = None,
        psnr_base: float = 10.0,
        image_size: Optional[Tuple[int, int]] = (512, 512),
        name: str = "peak_signal_noise_ratio",
    ) -> None:
        super().__init__(name)
        self.image_size = image_size
        self.psnr_metric = partial(
            peak_signal_noise_ratio, data_range=psnr_data_range, base=psnr_base
        )
        self.config = {
            "psnr_base": psnr_base,
            "psnr_data_range": psnr_data_range,
            "image_size": image_size,
        }

    @weave.op()
    def compute_metric(
        self,
        ground_truth_pil_image: Image.Image,
        generated_pil_image: Image.Image,
        prompt: str,
    ) -> ComputeMetricOutput:
        ground_truth_image = torch.from_numpy(
            np.expand_dims(
                np.array(ground_truth_pil_image.resize(self.image_size)), axis=0
            ).astype(np.uint8)
        ).float()
        generated_image = torch.from_numpy(
            np.expand_dims(
                np.array(generated_pil_image.resize(self.image_size)), axis=0
            ).astype(np.uint8)
        ).float()
        return ComputeMetricOutput(
            score=float(self.psnr_metric(generated_image, ground_truth_image).detach()),
            ground_truth_image=base64_encode_image(ground_truth_pil_image),
        )

    @weave.op()
    def evaluate(
        self,
        prompt: str,
        ground_truth_image: str,
        model_output: Dict[str, Any],
        metadata: weave.Model,
    ) -> Union[float, Dict[str, float]]:
        _ = "PSNRMetric"
        return super().evaluate(prompt, ground_truth_image, model_output, metadata)

    @weave.op()
    async def evaluate_async(
        self,
        prompt: str,
        ground_truth_image: str,
        model_output: Dict[str, Any],
        metadata: weave.Model,
    ) -> Union[float, Dict[str, float]]:
        _ = "PSNRMetric"
        return self.evaluate(prompt, ground_truth_image, model_output, metadata)
