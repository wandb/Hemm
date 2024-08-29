import unittest

import wandb
import weave

from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.vqa import MultiModalLLMEvaluationMetric
from hemm.metrics.vqa.judges.mmllm_judges import OpenAIJudge, PromptCategory


class TestMultiModalLLMEvaluation(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        wandb.init(
            project="unit-tests",
            entity="hemm-eval",
            job_type="test_multimodal_llm_evaluation",
        )
        weave.init(project_name="hemm-eval/unit-tests")

    def test_multimodal_llm_evaluation(self):
        model = BaseDiffusionModel(
            diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
            enable_cpu_offfload=False,
            image_height=1024,
            image_width=1024,
        )
        evaluation_pipeline = EvaluationPipeline(model=model)

        judge = OpenAIJudge(prompt_property=PromptCategory.complex)
        metric = MultiModalLLMEvaluationMetric(judge=judge)
        evaluation_pipeline.add_metric(metric)

        evaluation_pipeline(
            dataset=[
                {"prompt": "The fluffy pillow was on the left of the striped blanket."},
                {"prompt": "The round clock was mounted on the white wall."},
                {"prompt": "The black chair is on the right of the wooden table."},
            ]
        )
