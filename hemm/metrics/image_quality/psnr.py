from functools import partial
from typing import Dict, Optional, Tuple, Union
from PIL import Image

import numpy as np
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio

from .base import BaseImageQualityMetric


class PSNRMetric(BaseImageQualityMetric):

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
        ).float()
        generated_image = torch.from_numpy(
            np.expand_dims(
                np.array(generated_pil_image.resize(self.image_size)), axis=0
            ).astype(np.uint8)
        ).float()
        return float(self.psnr_metric(generated_image, ground_truth_image).detach())