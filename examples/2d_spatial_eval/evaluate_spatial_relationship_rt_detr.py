from typing import Optional, Tuple

import fire
import weave

import wandb
from hemm.eval_pipelines import EvaluationPipeline
from hemm.metrics.spatial_relationship import SpatialRelationshipMetric2D
from hemm.metrics.spatial_relationship.judges import RTDETRSpatialRelationShipJudge
from hemm.models import BaseDiffusionModel


def main(
    project="2d-spatial-relationship",
    entity="hemm-eval",
    dataset_ref: Optional[str] = "2d-spatial-prompts-mscoco:v0",
    dataset_limit: Optional[int] = None,
    diffusion_model_address: str = "stabilityai/stable-diffusion-2-1",
    diffusion_model_enable_cpu_offfload: bool = False,
    image_size: Tuple[int, int] = (1024, 1024),
    detr_model_address: str = "PekingU/rtdetr_r50vd",
):
    wandb.init(project=project, entity=entity, job_type="evaluation")
    weave.init(project_name=f"{entity}/{project}")

    dataset = weave.ref(dataset_ref).get()
    dataset = dataset.rows[:dataset_limit] if dataset_limit else dataset

    diffusion_model = BaseDiffusionModel(
        diffusion_model_name_or_path=diffusion_model_address,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
        image_height=image_size[0],
        image_width=image_size[1],
    )
    evaluation_pipeline = EvaluationPipeline(model=diffusion_model)

    judge = RTDETRSpatialRelationShipJudge(model_address=detr_model_address)
    metric = SpatialRelationshipMetric2D(
        judge=judge, name="2d_spatial_relationship_score"
    )
    evaluation_pipeline.add_metric(metric)

    evaluation_pipeline(dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
