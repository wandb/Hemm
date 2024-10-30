import asyncio
from abc import ABC
from typing import Dict, List, Union

import weave


class EvaluationPipeline(ABC):
    """Evaluation pipeline to evaluate the a multi-modal generative model.

    Args:
        model (BaseDiffusionModel): The model to evaluate.
    """

    def __init__(self, model: weave.Model) -> None:
        super().__init__()
        self.model = model

        self.image_size = (self.model.image_height, self.model.image_width)
        self.scorers = []

    def add_metric(self, metric: Union[callable, weave.Scorer]):
        """Add a metric function to the evaluation pipeline.

        Args:
            metric (BaseMetric): Metric function to evaluate the generated images.
        """
        self.scorers.append(metric)

    def __call__(self, dataset: Union[List[Dict], str]) -> Dict[str, float]:
        """Evaluate the Stable Diffusion model on the given dataset.

        Args:
            dataset (Union[List[Dict], str]): Dataset to evaluate the model on. If a string is
                passed, it is assumed to be a Weave dataset reference.
        """
        dataset = weave.ref(dataset).get() if isinstance(dataset, str) else dataset
        evaluation = weave.Evaluation(dataset=dataset, scorers=self.scorers)
        summary = asyncio.run(evaluation.evaluate())
        return summary
