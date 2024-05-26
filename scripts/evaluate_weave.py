from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.prompt_alignment import (
    BLIPScorer,
    CLIPImageQualityScorer,
    CLIPScorer,
)


if __name__ == "__main__":
    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        "CompVis/stable-diffusion-v1-4"
    )

    # Add CLIP Scorer metric
    clip_scorer = CLIPScorer()
    diffuion_evaluation_pipeline.add_metric(clip_scorer)

    # Add CLIP IQA Metric
    clip_iqa_scorer = CLIPImageQualityScorer()
    diffuion_evaluation_pipeline.add_metric(clip_iqa_scorer)

    diffuion_evaluation_pipeline(
        dataset="parti-prompts:v1",
        init_params=dict(project="t2i_eval", entity="geekyrakshit"),
    )
