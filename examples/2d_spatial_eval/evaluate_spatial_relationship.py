import os

import fire
import jsonlines
import rich
import wandb
import weave

from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.spatial_relationship import SpatialRelationshipMetric2D
from hemm.metrics.spatial_relationship.judges import DETRSpatialRelationShipJudge


def main():
    spatial_prompt_file = os.path.join(
        "/home/ubuntu/Hemm/artifacts/t2i_compbench_spatial_prompts:v1", "spatial.jsonl"
    )
    with jsonlines.open(spatial_prompt_file) as reader:
        for obj in reader:
            dataset = obj[:5]

    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        "stabilityai/stable-diffusion-2-1"
    )

    judge = DETRSpatialRelationShipJudge()
    metric = SpatialRelationshipMetric2D(judge=judge)
    diffuion_evaluation_pipeline.add_metric(metric)

    diffuion_evaluation_pipeline(
        dataset=dataset,
        init_params=dict(project="2d-spatial-relationship", entity="hemm-eval"),
        job_name="test",
    )


if __name__ == "__main__":
    main()
