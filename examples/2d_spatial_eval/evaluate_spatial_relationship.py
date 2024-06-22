import os
from typing import Optional, Tuple

import fire
import jsonlines
import weave

import wandb
from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.spatial_relationship import SpatialRelationshipMetric2D
from hemm.metrics.spatial_relationship.judges import DETRSpatialRelationShipJudge


def main(
    project="2d-spatial-relationship",
    entity="hemm-eval",
    dataset_artifact: str = "hemm-eval/2d-spatial-relationship/2d-spatial-prompts-mscoco:v0",
    dataset_ref: Optional[str] = "2d-spatial-prompts-mscoco:v0",
    dataset_limit: Optional[int] = None,
    diffusion_model_address: str = "stabilityai/stable-diffusion-2-1",
    diffusion_model_enable_cpu_offfload: bool = False,
    image_size: Tuple[int, int] = (512, 512),
    detr_model_address: str = "facebook/detr-resnet-50",
    detr_revision: str = "no_timm",
):
    wandb.init(project=project, entity=entity, job_type="evaluation")
    weave.init(project_name=project)
    artifact = wandb.use_artifact(dataset_artifact, type="dataset")
    artifact_dir = artifact.download()

    if dataset_limit:
        spatial_prompt_file = os.path.join(artifact_dir, "dataset.jsonl")
        with jsonlines.open(spatial_prompt_file) as reader:
            for obj in reader:
                dataset = obj[:dataset_limit]
    else:
        dataset = weave.ref(dataset_ref).get()

    diffusion_model = BaseDiffusionModel(
        diffusion_model_name_or_path=diffusion_model_address,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
        image_height=image_size[0],
        image_width=image_size[1],
    )
    evaluation_pipeline = EvaluationPipeline(model=diffusion_model)

    judge = DETRSpatialRelationShipJudge(
        model_address=detr_model_address, revision=detr_revision
    )
    metric = SpatialRelationshipMetric2D(
        judge=judge, name="2d_spatial_relationship_score"
    )
    evaluation_pipeline.add_metric(metric)

    evaluation_pipeline(dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
