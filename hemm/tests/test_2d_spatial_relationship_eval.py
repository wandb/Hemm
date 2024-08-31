import unittest

import wandb
import weave

from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.spatial_relationship import SpatialRelationshipMetric2D
from hemm.metrics.spatial_relationship.judges import (
    DETRSpatialRelationShipJudge,
    RTDETRSpatialRelationShipJudge,
)


class Test2DSpatialRelationshipEval(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        wandb.init(
            project="unit-tests",
            entity="hemm-eval",
            job_type="test_2d_spatial_relationship_evaluation",
        )
        weave.init(project_name="hemm-eval/unit-tests")

    def test_2d_spatial_relationship_evaluation_detr_judge(self):
        model = BaseDiffusionModel(
            diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
            enable_cpu_offfload=False,
        )
        evaluation_pipeline = EvaluationPipeline(model=model)

        judge = DETRSpatialRelationShipJudge(
            model_address="facebook/detr-resnet-50", revision="no_timm"
        )
        metric = SpatialRelationshipMetric2D(
            judge=judge, name="2d_spatial_relationship_score"
        )
        evaluation_pipeline.add_metric(metric)

        dataset = weave.ref("2d-spatial-prompts-mscoco:v0").get().rows[:2]
        summary = evaluation_pipeline(dataset=dataset)

        self.assertGreater(
            summary["SpatialRelationshipMetric2D.evaluate_async"][
                "2d_spatial_relationship_score"
            ]["mean"],
            0.0,
        )
        self.assertGreater(summary["model_latency"]["mean"], 0.0)

    def test_2d_spatial_relationship_evaluation_rt_detr_judge(self):
        model = BaseDiffusionModel(
            diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
            enable_cpu_offfload=False,
        )
        evaluation_pipeline = EvaluationPipeline(model=model)

        judge = RTDETRSpatialRelationShipJudge(model_address="PekingU/rtdetr_r50vd")
        metric = SpatialRelationshipMetric2D(
            judge=judge, name="2d_spatial_relationship_score"
        )
        evaluation_pipeline.add_metric(metric)

        dataset = weave.ref("2d-spatial-prompts-mscoco:v0").get().rows[:2]
        summary = evaluation_pipeline(dataset=dataset)

        self.assertGreater(
            summary["SpatialRelationshipMetric2D.evaluate_async"][
                "2d_spatial_relationship_score"
            ]["mean"],
            0.0,
        )
        self.assertGreater(summary["model_latency"]["mean"], 0.0)
