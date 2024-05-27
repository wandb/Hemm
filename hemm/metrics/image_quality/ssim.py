from functools import partial
from typing import Dict, Optional, Tuple, Union
from PIL import Image

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from .base import BaseImageQualityMetric


class SSIMMetric(BaseImageQualityMetric):

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

    def compute_metric(
        self,
        ground_truth_pil_image: Image.Image,
        generated_pil_image: Image.Image,
        prompt: str,
    ) -> Union[float, Dict[str, float]]:
        ground_truth_image = torch.from_numpy(
            np.expand_dims(
                np.array(ground_truth_pil_image.resize(self.image_size)), axis=0
            ).astype(np.uint8)
        ).permute(0, 3, 1, 2).float()
        generated_image = torch.from_numpy(
            np.expand_dims(
                np.array(generated_pil_image.resize(self.image_size)), axis=0
            ).astype(np.uint8)
        ).permute(0, 3, 1, 2).float()
        return float(self.ssim_metric(generated_image, ground_truth_image))
