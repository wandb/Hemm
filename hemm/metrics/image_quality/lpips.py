from functools import partial
from typing import Dict, Literal, Optional, Tuple, Union
from PIL import Image

import numpy as np
import torch
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity

from .base import BaseImageQualityMetric


class LPIPSMetric(BaseImageQualityMetric):

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

    def compute_metric(
        self, ground_truth_pil_image: Image, generated_pil_image: Image, prompt: str
    ) -> Union[float, Dict[str, float]]:
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
        return float(self.lpips_metric(generated_image, ground_truth_image).detach())
