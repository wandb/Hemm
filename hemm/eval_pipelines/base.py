import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

import wandb
import weave
from weave import Evaluation

from ..utils import base64_encode_image


class BaseEvaluationPipeline(ABC):
    """Base class for evaluation pipelines.

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
        super().__init__()
        self.diffusion_model_name_or_path = diffusion_model_name_or_path
        self.torch_dtype = torch_dtype
        self.enable_cpu_offfload = enable_cpu_offfload
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        self.seed = seed

        self.inference_counter = 1
        self.table_columns = ["model", "prompt", "generated_image"]
        self.table_rows: List = []
        self.wandb_table: wandb.Table = None
        self.metric_functions: List[Callable] = []

        self.evaluation_configs = {
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
        self.inference_counter += 1
        image = self.pipeline(
            prompt,
            num_images_per_prompt=1,
            height=self.image_size[0],
            width=self.image_size[1],
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        ).images[0]
        self.table_rows.append(
            [self.diffusion_model_name_or_path, prompt, wandb.Image(image)]
        )
        return {"image": base64_encode_image(image)}

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
