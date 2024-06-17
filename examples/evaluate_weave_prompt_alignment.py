import fire
import wandb
import weave

from hemm.eval_pipelines import BaseWeaveModel, EvaluationPipeline
from hemm.metrics.prompt_alignment import CLIPScoreMetric, CLIPImageQualityScoreMetric


def main(
    diffusion_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
    clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
    clip_iqa_model_name_or_path: str = "clip_iqa",
    diffusion_model_enable_cpu_offfload: bool = False,
    dataset: str = "parti-prompts:v0",
    project: str = "propmpt-alignment",
):
    wandb.init(project=project, job_type="evaluation")
    weave.init(project_name=project)

    model = BaseWeaveModel(
        diffusion_model_name_or_path=diffusion_model_name_or_path,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
    )
    evaluation_pipeline = EvaluationPipeline(model=model)

    # Add CLIP Scorer metric
    clip_scorer = CLIPScoreMetric(clip_model_name_or_path=clip_model_name_or_path)
    evaluation_pipeline.add_metric(clip_scorer)

    # Add CLIP IQA Metric
    clip_iqa_scorer = CLIPImageQualityScoreMetric(
        clip_model_name_or_path=clip_iqa_model_name_or_path
    )
    evaluation_pipeline.add_metric(clip_iqa_scorer)

    evaluation_pipeline(dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
