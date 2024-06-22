import asyncio
from abc import ABC
from typing import Dict, List, Union

import weave

import wandb

from ..metrics.base import BaseMetric
from ..utils import base64_decode_image
from .hemm_evaluation import AsyncHemmEvaluation, SyncHemmEvaluation
from .model import BaseDiffusionModel


class EvaluationPipeline(ABC):
    """Evaluation pipeline to evaluate the a multi-modal generative model.

    Args:
        model (BaseDiffusionModel): The model to evaluate.
        seed (int): Seed value for the random number generator.
    """

    def __init__(self, model: BaseDiffusionModel, seed: int = 42) -> None:
        super().__init__()
        self.model = model
        self.model.initialize()

        self.image_size = (self.model.image_height, self.model.image_width)
        self.seed = seed

        self.inference_counter = 1
        self.table_columns = ["model", "prompt", "generated_image"]
        self.table_rows: List = []
        self.wandb_table: wandb.Table = None
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
            self.wandb_table = wandb.Table(columns=self.table_columns)
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

    def log_summary(self):
        """Log the evaluation summary to the Weights & Biases dashboard."""
        config = wandb.config
        config.update(self.evaluation_configs)
        for row_idx, row in enumerate(self.table_rows):
            current_row = row
            for metric_fn in self.metric_functions:
                current_row.append(metric_fn.scores[row_idx])
            self.wandb_table.add_data(*current_row)
        wandb.log(
            {f"Evalution/{self.model.diffusion_model_name_or_path}": self.wandb_table}
        )

    def __call__(
        self, dataset: Union[List[Dict], str], async_evaluation: bool = False
    ) -> None:
        """Evaluate the Stable Diffusion model on the given dataset.

        Args:
            dataset (Union[List[Dict], str]): Dataset to evaluate the model on. If a string is
                passed, it is assumed to be a Weave dataset reference.
            async_evaluation (bool): Flag to enable asynchronous evaluation.
        """
        dataset = weave.ref(dataset).get() if isinstance(dataset, str) else dataset
        if async_evaluation:
            evaluation = AsyncHemmEvaluation(
                dataset=dataset,
                scorers=[
                    metric_fn.evaluate_async for metric_fn in self.metric_functions
                ],
            )
            with weave.attributes(self.evaluation_configs):
                asyncio.run(evaluation.evaluate(self.infer_async))
        else:
            evaluation = SyncHemmEvaluation(
                dataset=dataset,
                scorers=[metric_fn.evaluate for metric_fn in self.metric_functions],
            )
            with weave.attributes(self.evaluation_configs):
                evaluation.evaluate(self.infer)
        self.log_summary()
