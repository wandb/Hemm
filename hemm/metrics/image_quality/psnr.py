from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import weave
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio


class PSNRMetric(weave.Scorer):
    """PSNR Metric to compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        psnr_base (float): The base of the logarithm in the PSNR formula.
        psnr_data_range (Optional[Union[float, Tuple[float, float]]]): The data range of the input
            image (min, max). If None, the data range is determined from the image data type.
        image_height (int): The height to which images will be resized before computing PSNR.
        image_width (int): The width to which images will be resized before computing PSNR.
    """

    psnr_base: float
    psnr_data_range: Optional[Union[float, Tuple[float, float]]]
    image_height: int
    image_width: int
    _psnr_metric: Callable

    def __init__(
        self,
        psnr_data_range: Optional[Union[float, Tuple[float, float]]] = None,
        psnr_base: float = 10.0,
        image_height: int = 512,
        image_width: int = 512,
    ) -> None:
        super().__init__(
            psnr_data_range=psnr_data_range,
            psnr_base=psnr_base,
            image_height=image_height,
            image_width=image_width,
        )
        self._psnr_metric = partial(
            peak_signal_noise_ratio,
            data_range=self.psnr_data_range,
            base=self.psnr_base,
        )

    @weave.op()
    def compute_metric(
        self, ground_truth_pil_image: Image.Image, generated_pil_image: Image.Image
    ) -> Dict[str, float]:
        ground_truth_image = torch.from_numpy(
            np.expand_dims(
                np.array(
                    ground_truth_pil_image.resize((self.image_height, self.image_width))
                ),
                axis=0,
            ).astype(np.uint8)
        ).float()
        generated_image = torch.from_numpy(
            np.expand_dims(
                np.array(
                    generated_pil_image.resize((self.image_height, self.image_width))
                ),
                axis=0,
            ).astype(np.uint8)
        ).float()
        return {
            "score": float(
                self._psnr_metric(generated_image, ground_truth_image).detach()
            ),
            "ground_truth_image": ground_truth_pil_image,
        }

    @weave.op()
    def score(
        self, prompt: str, ground_truth_image: Image.Image, model_output: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        _ = prompt
        metric_output = self.compute_metric(ground_truth_image, model_output["image"])
        return {"score": metric_output["score"]}
