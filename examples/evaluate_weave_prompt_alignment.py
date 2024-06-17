import fire

from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.prompt_alignment import CLIPScoreMetric, CLIPImageQualityScoreMetric


def main(
    diffusion_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
    clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
    clip_iqa_model_name_or_path: str = "clip_iqa",
    dataset: str = "parti-prompts:v0",
    project: str = "propmpt-alignment",
):
    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        diffusion_model_name_or_path
    )

    # Add CLIP Scorer metric
    clip_scorer = CLIPScoreMetric(clip_model_name_or_path=clip_model_name_or_path)
    diffuion_evaluation_pipeline.add_metric(clip_scorer)

    # Add CLIP IQA Metric
    clip_iqa_scorer = CLIPImageQualityScoreMetric(
        clip_model_name_or_path=clip_iqa_model_name_or_path
    )
    diffuion_evaluation_pipeline.add_metric(clip_iqa_scorer)

    diffuion_evaluation_pipeline(
        dataset=dataset,
        init_params=dict(project=project),
        job_name="test",
    )


if __name__ == "__main__":
    fire.Fire(main)
