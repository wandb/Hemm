import asyncio
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import StableDiffusionPipeline

import wandb
import weave
from weave import Evaluation

from ..utils import base64_encode_image


class StableDiffusionEvaluationPipeline:
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
        images_dir (str): Directory to save the generated images.
    """

    def __init__(
        self,
        diffusion_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float16,
        enable_cpu_offfload: bool = False,
        image_size: Optional[Union[int, Tuple[int, int]]] = 512,
        seed: int = 42,
        images_dir: str = "generated_images",
    ) -> None:
        self.diffusion_model_name_or_path = diffusion_model_name_or_path
        self.torch_dtype = torch_dtype
        self.enable_cpu_offfload = enable_cpu_offfload
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        self.seed = seed

        self.generated_images_dir = os.path.join(os.getcwd(), images_dir)
        os.makedirs(self.generated_images_dir, exist_ok=True)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            diffusion_model_name_or_path, torch_dtype=torch_dtype
        )

        if enable_cpu_offfload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(leave=False, desc="Generating Image")

        self.inference_counter = 1
        self.table_columns = ["model", "prompt", "generated_image"]
        self.table_rows: List = []
        self.wandb_table: wandb.Table = None
        self.metric_functions: List[Callable] = []

        self.evaluation_configs = {
            "diffusion_pipeline": dict(self.pipeline.config),
            "torch_dtype": str(torch_dtype),
            "enable_cpu_offfload": enable_cpu_offfload,
            "image_size": {
                "height": self.image_size[0],
                "width": self.image_size[1],
            },
            "seed": seed,
        }

    def add_metric(self, metric_fn: Callable):
        """Add a metric function to the evaluation pipeline.
        
        Args:
            metric_fn (Callable): Metric function to evaluate the generated images.
        """
        self.table_columns.append(metric_fn.__class__.__name__)
        self.evaluation_configs.update(metric_fn.config)
        self.metric_functions.append(metric_fn)

    @weave.op()
    async def infer(self, prompt: str) -> Dict[str, str]:
        """Inference function to generate images for the given prompt.
        
        Args:
            prompt (str): Prompt to generate the image.
        
        Returns:
            Dict[str, str]: Dictionary containing base64 encoded image to be logged as
                a Weave object.
        """
        if self.inference_counter == 1:
            self.wandb_table = wandb.Table(columns=self.table_columns)
        image_path = os.path.join(
            self.generated_images_dir, f"{self.inference_counter}.png"
        )
        self.inference_counter += 1
        self.pipeline(
            prompt,
            num_images_per_prompt=1,
            height=self.image_size[0],
            width=self.image_size[1],
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        ).images[0].save(image_path)
        self.table_rows.append(
            [self.diffusion_model_name_or_path, prompt, wandb.Image(image_path)]
        )
        return {"image": base64_encode_image(image_path)}

    def log_summary(self, init_params: Dict):
        """Log the evaluation summary to the Weights & Biases dashboard.
        
        Args:
            init_params (Dict): Initialization parameters for the Weights & Biases run. Refer
                to [official docs](https://docs.wandb.ai/ref/python/init) for more documentation
                on the initialization params.
        """
        if wandb.run is None:
            wandb.init(**init_params)
        config = wandb.config
        config.update(self.evaluation_configs)
        for row_idx, row in enumerate(self.table_rows):
            current_row = row
            for metric_fn in self.metric_functions:
                current_row.append(metric_fn.scores[row_idx])
            self.wandb_table.add_data(*current_row)
        wandb.log({"Evalution": self.wandb_table})

    def __call__(self, dataset: Union[Dict, str], init_params: Dict):
        """Evaluate the Stable Diffusion model on the given dataset.
        
        Args:
            dataset (Union[Dict, str]): Dataset to evaluate the model on. If a string is passed,
                it is assumed to be a Weave dataset reference.
            init_params (Dict): Initialization parameters for the Weights & Biases run. Refer
                to [official docs](https://docs.wandb.ai/ref/python/init) for more documentation
                on the initialization params.
        """
        weave.init(project_name="t2i_eval")
        dataset = weave.ref(dataset).get() if isinstance(dataset, str) else dataset
        evaluation = Evaluation(
            dataset=dataset,
            scorers=[metric_fn.__call__ for metric_fn in self.metric_functions],
        )
        with weave.attributes(self.evaluation_configs):
            asyncio.run(evaluation.evaluate(self.infer))
        self.log_summary(init_params)
