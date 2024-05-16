import asyncio
import os
from typing import Callable, Dict, List

import torch
from diffusers import StableDiffusionPipeline

import wandb
import weave
from weave import Evaluation

from ..utils import image_to_data_url


class StableDiffusionEvaluationPipeline:

    def __init__(
        self,
        diffusion_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float16,
        enable_cpu_offfload: bool = False,
        seed: int = 42,
        images_dir: str = "generated_images",
    ) -> None:
        self.diffusion_model_name_or_path = diffusion_model_name_or_path
        self.torch_dtype = torch_dtype
        self.enable_cpu_offfload = enable_cpu_offfload
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
        self.table_columns = ["model", "prompt", "image"]
        self.table_rows: List = []
        self.wandb_table: wandb.Table = None
        self.metric_functions: List[Callable] = []

        self.evaluation_configs = {
            "diffusion_pipeline": dict(self.pipeline.config),
            "torch_dtype": str(torch_dtype),
            "enable_cpu_offfload": enable_cpu_offfload,
            "seed": seed,
        }

    def add_metric(self, metric_fn: Callable):
        self.table_columns.append(metric_fn.__class__.__name__)
        self.evaluation_configs.update(metric_fn.config)
        self.metric_functions.append(metric_fn)

    @weave.op()
    async def infer(self, prompt: str) -> Dict[str, str]:
        if self.inference_counter == 1:
            self.wandb_table = wandb.Table(columns=self.table_columns)
        image_path = os.path.join(
            self.generated_images_dir, f"{self.inference_counter}.png"
        )
        self.inference_counter += 1
        self.pipeline(
            prompt,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        ).images[0].save(image_path)
        self.table_rows.append(
            [self.diffusion_model_name_or_path, prompt, wandb.Image(image_path)]
        )
        return {"image": image_to_data_url(image_path)}

    def log_summary(self, init_params: Dict):
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

    def __call__(self, dataset: Dict, init_params: Dict):
        weave.init(project_name="t2i_eval")
        evaluation = Evaluation(
            dataset=dataset,
            scorers=[metric_fn.__call__ for metric_fn in self.metric_functions],
        )
        with weave.attributes(self.evaluation_configs):
            asyncio.run(evaluation.evaluate(self.infer))
        self.log_summary(init_params)
