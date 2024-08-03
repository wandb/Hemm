import asyncio
from abc import ABC
from functools import partial
from typing import Any, Dict, List, Optional, Union

import wandb
import weave

from ..metrics.base import BaseMetric
from ..utils import base64_decode_image
from .model import BaseDiffusionModel


class EvaluationPipeline(ABC):
    """Evaluation pipeline to evaluate the a multi-modal generative model.

    Args:
        model (BaseDiffusionModel): The model to evaluate.
        seed (int): Seed value for the random number generator.
    """

    def __init__(
        self,
        model: BaseDiffusionModel,
        seed: int = 42,
        configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.initialize()

        self.image_size = (self.model.image_height, self.model.image_width)
        self.seed = seed

        self.inference_counter = 1
        self.table_columns = ["model", "prompt", "generated_image"]
        self.table_rows: List = []
        self.evaluation_table: wandb.Table = None
        self.metric_functions: List[BaseMetric] = []

        self.evaluation_configs = {
            "pretrained_model_name_or_path": self.model.diffusion_model_name_or_path,
            "torch_dtype": str(self.model._torch_dtype),
            "enable_cpu_offfload": self.model.enable_cpu_offfload,
            "image_size": {
                "height": self.image_size[0],
                "width": self.image_size[1],
            },
            "seed": seed,
            "diffusion_pipeline": dict(self.model._pipeline.config),
            **configs,
        }

    def add_metric(self, metric_fn: BaseMetric):
        """Add a metric function to the evaluation pipeline.

        Args:
            metric_fn (BaseMetric): Metric function to evaluate the generated images.
        """
        self.table_columns.append(metric_fn.__class__.__name__)
        self.evaluation_configs.update(metric_fn.config)
        self.metric_functions.append(metric_fn)

    @weave.op()
    def infer(self, prompt: str) -> Dict[str, str]:
        """Inference function to generate images for the given prompt.

        Args:
            prompt (str): Prompt to generate the image.

        Returns:
            Dict[str, str]: Dictionary containing base64 encoded image to be logged as
                a Weave object.
        """
        if self.inference_counter == 1:
            self.evaluation_table = wandb.Table(columns=self.table_columns)
        self.inference_counter += 1
        output = self.model.predict(prompt, seed=self.seed)
        self.table_rows.append(
            [
                self.model.diffusion_model_name_or_path,
                prompt,
                wandb.Image(base64_decode_image(output["image"])),
            ]
        )
        return output

    @weave.op()
    async def infer_async(self, prompt: str) -> Dict[str, str]:
        """Async inference function to generate images for the given prompt.

        Args:
            prompt (str): Prompt to generate the image.

        Returns:
            Dict[str, str]: Dictionary containing base64 encoded image to be logged as
                a Weave object.
        """
        return self.infer(prompt)

    def log_summary(self, summary: Dict[str, float]) -> None:
        """Log the evaluation summary to the Weights & Biases dashboard."""
        config = wandb.config
        config.update(self.evaluation_configs)
        for row_idx, row in enumerate(self.table_rows):
            current_row = row
            for metric_fn in self.metric_functions:
                current_row.append(metric_fn.scores[row_idx])
            self.evaluation_table.add_data(*current_row)
        summary_table = wandb.Table(columns=["summary"], data=[[summary]])
        wandb.log(
            {
                f"evalution/{self.model.diffusion_model_name_or_path}": self.evaluation_table,
                f"summary/{self.model.diffusion_model_name_or_path}": summary_table,
                "summary": summary,
            }
        )

    @weave.op()
    def evaluate(
        self, dataset: Union[List[Dict], str], name: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate the Stable Diffusion model on the given dataset.

        Args:
            dataset (Union[List[Dict], str]): Dataset to evaluate the model on. If a string is
                passed, it is assumed to be a Weave dataset reference.
        """
        dataset = weave.ref(dataset).get() if isinstance(dataset, str) else dataset
        evaluation = weave.Evaluation(
            name=name,
            dataset=dataset,
            scorers=[
                partial(
                    metric_fn.evaluate_async,
                    metadata=weave.Model.model_validate(self.evaluation_configs),
                )
                for metric_fn in self.metric_functions
            ],
        )
        summary = asyncio.run(evaluation.evaluate(self.infer_async))
        self.log_summary(summary)
        return summary
