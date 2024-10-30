import asyncio

import weave

from hemm.metrics.spatial_relationship import SpatialRelationshipMetric2D
from hemm.metrics.spatial_relationship.judges import (
    DETRSpatialRelationShipJudge,
    RTDETRSpatialRelationShipJudge,
)
from hemm.models import DiffusersModel


def test_2d_spatial_relationship_evaluation_detr_judge():
    weave.init(project_name="hemm-eval/unit-tests")
    model = DiffusersModel(
        diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
        enable_cpu_offfload=False,
    )

    judge = DETRSpatialRelationShipJudge(
        model_address="facebook/detr-resnet-50", revision="no_timm"
    )
    metric = SpatialRelationshipMetric2D(
        judge=judge, name="2d_spatial_relationship_score"
    )

    dataset = weave.ref("2d-spatial-prompts-mscoco:v0").get().rows[:2]
    evaluation = weave.Evaluation(dataset=dataset, scorers=[metric])
    summary = asyncio.run(evaluation.evaluate(model))

    assert (
        summary["SpatialRelationshipMetric2D.evaluate_async"][
            "2d_spatial_relationship_score"
        ]["mean"]
        > 0
    )
    assert summary["model_latency"]["mean"] > 0


def test_2d_spatial_relationship_evaluation_rt_detr_judge():
    weave.init(project_name="hemm-eval/unit-tests")
    model = DiffusersModel(
        diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
        enable_cpu_offfload=False,
    )

    judge = RTDETRSpatialRelationShipJudge(model_address="PekingU/rtdetr_r50vd")
    metric = SpatialRelationshipMetric2D(
        judge=judge, name="2d_spatial_relationship_score"
    )

    dataset = weave.ref("2d-spatial-prompts-mscoco:v0").get().rows[:2]
    evaluation = weave.Evaluation(dataset=dataset, scorers=[metric])
    summary = asyncio.run(evaluation.evaluate(model))

    assert (
        summary["SpatialRelationshipMetric2D.evaluate_async"][
            "2d_spatial_relationship_score"
        ]["mean"]
        > 0
    )
    assert summary["model_latency"]["mean"] > 0
