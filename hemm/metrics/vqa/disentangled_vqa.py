from typing import Any, Dict, Optional, Union

import weave

from ..base import BaseMetric
from .judges import BlipVQAJudge


class DisentangledVQAMetric(BaseMetric):
    """Disentangled VQA metric to evaluate the attribute-binding capability
    for image generation models as proposed in Section 4.1 from the paper
    [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/pdf/2307.06350).

    ??? example "Sample usage"
        ```python
        import wandb
        import weave
        from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
        from hemm.metrics.vqa import DisentangledVQAMetric
        from hemm.metrics.vqa.judges import BlipVQAJudge

        wandb.init(project=project, entity=entity, job_type="evaluation")
        weave.init(project_name=project)

        diffusion_model = BaseDiffusionModel(
            diffusion_model_name_or_path=diffusion_model_address,
            enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
            image_height=image_size[0],
            image_width=image_size[1],
        )
        evaluation_pipeline = EvaluationPipeline(model=diffusion_model)

        judge = BlipVQAJudge()
        metric = DisentangledVQAMetric(judge=judge, name="disentangled_blip_metric")
        evaluation_pipeline.add_metric(metric)

        evaluation_pipeline(dataset=dataset)
        ```

    Args:
        judge (Union[weave.Model, BlipVQAJudge]): The judge model to evaluate the attribute-binding capability.
        name (Optional[str]): The name of the metric. Defaults to "disentangled_vlm_metric".
    """

    def __init__(
        self,
        judge: Union[weave.Model, BlipVQAJudge],
        name: Optional[str] = "disentangled_vlm_metric",
    ) -> None:
        super().__init__()
        self.judge = judge
        self.config = self.judge.model_dump()
        self.scores = []
        self.name = name

    @weave.op()
    def evaluate(
        self,
        prompt: str,
        adj_1: str,
        noun_1: str,
        adj_2: str,
        noun_2: str,
        model_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate the attribute-binding capability of the model.

        Args:
            prompt (str): The prompt for the model.
            adj_1 (str): The first adjective.
            noun_1 (str): The first noun.
            adj_2 (str): The second adjective.
            noun_2 (str): The second noun.
            model_output (Dict[str, Any]): The model output.

        Returns:
            Dict[str, Any]: The evaluation result.
        """
        _ = prompt
        judgement = self.judge.predict(
            adj_1, noun_1, adj_2, noun_2, model_output["image"]
        )
        self.scores.append(judgement)
        return judgement

    @weave.op()
    async def evaluate_async(
        self,
        prompt: str,
        adj_1: str,
        noun_1: str,
        adj_2: str,
        noun_2: str,
        model_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.evaluate(prompt, adj_1, noun_1, adj_2, noun_2, model_output)
