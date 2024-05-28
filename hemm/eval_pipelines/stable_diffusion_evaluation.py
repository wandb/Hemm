from typing import Optional, Tuple, Union

import torch
from diffusers import StableDiffusionPipeline

from .base import BaseEvaluationPipeline


class StableDiffusionEvaluationPipeline(BaseEvaluationPipeline):
    """Evaluation pipeline for Stable Diffusion variant models using the
    [`diffusers.StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/v0.27.2/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline).

    ??? example "Evaluating Stable Diffusion v1.4 on prompt alignment metrics"
        ```python
        from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
        from hemm.metrics.prompt_alignment import CLIPScoreMetric, CLIPImageQualityScoreMetric


        if __name__ == "__main__":
            diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
                "CompVis/stable-diffusion-v1-4"
            )

            # Add CLIP Scorer metric
            clip_scorer = CLIPScoreMetric()
            diffuion_evaluation_pipeline.add_metric(clip_scorer)

            # Add CLIP IQA Metric
            clip_iqa_scorer = CLIPImageQualityScoreMetric()
            diffuion_evaluation_pipeline.add_metric(clip_iqa_scorer)

            diffuion_evaluation_pipeline(
                dataset="parti-prompts:v1",
                init_params=dict(project="t2i_eval", entity="geekyrakshit"),
            )
        ```

    ??? example "Evaluating Stable Diffusion v1.4 on image quality metrics"
        ```python
        from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
        from hemm.metrics.image_quality import LPIPSMetric, PSNRMetric, SSIMMetric


        if __name__ == "__main__":
            diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
                "CompVis/stable-diffusion-v1-4"
            )

            # Add PSNR Metric
            psnr_metric = PSNRMetric(image_size=diffuion_evaluation_pipeline.image_size)
            diffuion_evaluation_pipeline.add_metric(psnr_metric)

            # Add SSIM Metric
            ssim_metric = SSIMMetric(image_size=diffuion_evaluation_pipeline.image_size)
            diffuion_evaluation_pipeline.add_metric(ssim_metric)

            # Add LPIPS Metric
            lpips_metric = LPIPSMetric(image_size=diffuion_evaluation_pipeline.image_size)
            diffuion_evaluation_pipeline.add_metric(lpips_metric)

            diffuion_evaluation_pipeline(
                dataset="COCO:v1",
                init_params=dict(project="t2i_eval", entity="geekyrakshit"),
            )
        ```

    Args:
        diffusion_model_name_or_path (str): Name or path of the pre-trained Stable Diffusion
            variant model.
        torch_dtype (torch.dtype): Override the default `torch.dtype` and load the model with
            another dtype. If "auto" is passed, the dtype is automatically derived from the
            model's weights.
        enable_cpu_offfload (bool): Whether to enable CPU offloading for the model during
            evaluation.
        image_size (Optional[Union[int, Tuple[int, int]]]): Size of the generated image. If an
            integer is passed, the image will be resized to a square image of that size.
        seed (int): Seed value for the random number generator.
    """

    def __init__(
        self,
        diffusion_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float16,
        enable_cpu_offfload: bool = False,
        image_size: Optional[Union[int, Tuple[int, int]]] = 512,
        seed: int = 42,
    ) -> None:
        super().__init__(
            diffusion_model_name_or_path,
            torch_dtype,
            enable_cpu_offfload,
            image_size,
            seed,
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            diffusion_model_name_or_path, torch_dtype=torch_dtype
        )

        if enable_cpu_offfload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(leave=False, desc="Generating Image")

        self.evaluation_configs["diffusion_pipeline"] = dict(self.pipeline.config)
