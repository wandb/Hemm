import os
from typing import Optional, Tuple

import fire
import jsonlines
import wandb
import weave

from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.vqa import DisentangledVQAMetric
from hemm.metrics.vqa.judges import BlipVQAJudge


def main(
    project="disentangled_vqa",
    entity="hemm-eval",
    dataset_artifact: str = "hemm-eval/disentangled_vqa/attribute_binding_dataset:v1",
    dataset_ref: Optional[str] = "attribute_binding_dataset:v1",
    dataset_limit: Optional[int] = None,
    diffusion_model_address: str = "stabilityai/stable-diffusion-2-1",
    diffusion_model_enable_cpu_offfload: bool = False,
    image_size: Tuple[int, int] = (512, 512),
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

    judge = BlipVQAJudge()
    metric = DisentangledVQAMetric(judge=judge, name="disentangled_blip_metric")
    evaluation_pipeline.add_metric(metric)

    evaluation_pipeline(dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
