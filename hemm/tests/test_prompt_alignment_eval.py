import unittest

import wandb
import weave

from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.prompt_alignment import CLIPScoreMetric


class TestPromptAlignmentEvaluation(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        wandb.init(
            project="unit-tests",
            entity="hemm-eval",
            job_type="test_prompt_alignment_evaluation",
        )
        weave.init(project_name="hemm-eval/unit-tests")

    def test_prompt_alignment_evaluation(self):
        model = BaseDiffusionModel(
            diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
            enable_cpu_offfload=False,
        )
        evaluation_pipeline = EvaluationPipeline(model=model)

        # Add CLIP Scorer metric
        clip_scorer = CLIPScoreMetric(
            clip_model_name_or_path="openai/clip-vit-base-patch16"
        )
        evaluation_pipeline.add_metric(clip_scorer)

        dataset = weave.ref("parti-prompts:v0").get().rows[:2]
        summary = evaluation_pipeline(dataset=dataset)

        self.assertGreater(
            summary["CLIPScoreMetric.evaluate_async"]["clip_score"]["mean"], 0.0
        )
        self.assertGreater(summary["model_latency"]["mean"], 0.0)
