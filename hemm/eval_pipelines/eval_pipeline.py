import asyncio
import os
import shutil
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import weave
from PIL import Image

import wandb

from ..metrics.base import BaseMetric
from ..models import BaseDiffusionModel, FalAIModel, StabilityAPIModel

MODEL_TYPE = Union[BaseDiffusionModel, FalAIModel, StabilityAPIModel]


class EvaluationPipeline(ABC):
    """Evaluation pipeline to evaluate the a multi-modal generative model.

    Args:
        model (BaseDiffusionModel): The model to evaluate.
        seed (int): Seed value for the random number generator.
        save_inference_dataset_name (Optional[str]): A weave dataset name which if provided will
            save inference results as a separate weave dataset.
    """

    def __init__(
        self,
        model: MODEL_TYPE,
        seed: int = 42,
        save_inference_dataset_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model

        self.image_size = (self.model.image_height, self.model.image_width)
        self.seed = seed
        self.save_inference_dataset_name = save_inference_dataset_name

        if self.save_inference_dataset_name:
            os.makedirs(
                os.path.join("inference_dataset", self.save_inference_dataset_name),
                exist_ok=True,
            )

        self.inference_counter = 0
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
        if self.inference_counter == 0:
            self.evaluation_table = wandb.Table(columns=self.table_columns)
        output = self.model.predict(prompt, seed=self.seed)
        self.table_rows.append(
            [self.model.diffusion_model_name_or_path, prompt, output["image"]]
        )
        if self.save_inference_dataset_name:
            output["image"].save(
                os.path.join(
                    "inference_dataset",
                    self.save_inference_dataset_name,
                    f"{self.inference_counter}.png",
                )
            )
        self.inference_counter += 1
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
            current_row[-1] = wandb.Image(current_row[-1])
            for metric_fn in self.metric_functions:
                current_row.append(metric_fn.scores[row_idx])
            self.evaluation_table.add_data(*current_row)
        summary_table = wandb.Table(columns=["summary"], data=[[summary]])
        wandb.log(
            {
                "evalution": self.evaluation_table,
                "summary": summary_table,
                "evaluation_summary": summary,
            }
        )

    def save_inference_results(self, dataset: Any):
        inference_dataset_rows = []
        for idx, row in enumerate(dataset.rows):
            generated_image = Image.open(
                os.path.join(
                    "inference_dataset", self.save_inference_dataset_name, f"{idx}.png"
                )
            )
            inference_dataset_rows.append(
                {
                    "generated_image": generated_image,
                    "seed": self.seed,
                    **dict(row),
                    "image_generation_model": self.model.model_dump(),
                }
            )
        weave.publish(
            weave.Dataset(
                name=self.save_inference_dataset_name, rows=inference_dataset_rows
            )
        )
        shutil.rmtree("inference_dataset")

    def __call__(
        self, dataset: Union[List[Dict], str], async_infer: bool = False
    ) -> Dict[str, float]:
        """Evaluate the Stable Diffusion model on the given dataset.

        Args:
            dataset (Union[List[Dict], str]): Dataset to evaluate the model on. If a string is
                passed, it is assumed to be a Weave dataset reference.
            async_infer (bool, optional): Whether to use async inference. Defaults to False.
        """
        dataset = weave.ref(dataset).get() if isinstance(dataset, str) else dataset
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[
                metric_fn.evaluate_async if async_infer else metric_fn.evaluate
                for metric_fn in self.metric_functions
            ],
        )
        self.model.configs.update(self.evaluation_configs)
        summary = asyncio.run(evaluation.evaluate(self.infer_async))
        self.log_summary(summary)
        if self.save_inference_dataset_name:
            self.save_inference_results(dataset=dataset)
        return summary
