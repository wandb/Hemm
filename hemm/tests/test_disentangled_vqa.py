import unittest

import weave

import wandb
from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.vqa import DisentangledVQAMetric
from hemm.metrics.vqa.judges import BlipVQAJudge


class TestDisentangledVQA(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        wandb.init(
            project="unit-tests",
            entity="hemm-eval",
            job_type="test_desentangled_vqa_evaluation",
        )
        weave.init(project_name="hemm-eval/unit-tests")

    def test_desentangled_vqa_evaluation(self):
        model = BaseDiffusionModel(
            diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
            enable_cpu_offfload=False,
        )
        evaluation_pipeline = EvaluationPipeline(model=model)

        judge = BlipVQAJudge()
        metric = DisentangledVQAMetric(judge=judge, name="disentangled_blip_metric")
        evaluation_pipeline.add_metric(metric)

        dataset = weave.ref("attribute_binding_dataset:v0").get().rows[:2]
        summary = evaluation_pipeline(dataset=dataset)

        self.assertGreater(summary["model_latency"]["mean"], 0.0)
