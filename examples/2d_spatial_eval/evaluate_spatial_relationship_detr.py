import asyncio
from typing import Optional

import fire
import weave

from hemm.metrics.spatial_relationship import SpatialRelationshipMetric2D
from hemm.metrics.spatial_relationship.judges import DETRSpatialRelationShipJudge
from hemm.models import DiffusersModel


def main(
    project="2d-spatial-relationship",
    entity="hemm-eval",
    dataset_ref: Optional[str] = "2d-spatial-prompts-mscoco:v0",
    dataset_limit: Optional[int] = None,
    diffusion_model_address: str = "stabilityai/stable-diffusion-2-1",
    diffusion_model_enable_cpu_offfload: bool = False,
    image_height: int = 1024,
    image_width: int = 1024,
    detr_model_address: str = "facebook/detr-resnet-50",
    detr_revision: str = "no_timm",
    iou_threshold: Optional[float] = 0.1,
):
    weave.init(project_name=f"{entity}/{project}")

    dataset = weave.ref(dataset_ref).get()
    dataset = dataset.rows[:dataset_limit] if dataset_limit else dataset

    model = DiffusersModel(
        diffusion_model_name_or_path=diffusion_model_address,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
        image_height=image_height,
        image_width=image_width,
    )

    judge = DETRSpatialRelationShipJudge(
        model_address=detr_model_address, revision=detr_revision
    )
    metric = SpatialRelationshipMetric2D(judge=judge, iou_threshold=iou_threshold)

    evaluation = weave.Evaluation(dataset=dataset, scorers=[metric])
    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    fire.Fire(main)
