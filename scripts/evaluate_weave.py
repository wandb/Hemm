from t2i_diffusion_benchmark.metrics import (
    CLIPImageQualityScorer,
    CLIPScorer,
    BLIPScorer,
)
from t2i_diffusion_benchmark.eval_pipelines import StableDiffusionEvaluationPipeline


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

    # Add BLIP Scorer Metric
    blip_scorer = BLIPScorer()
    diffuion_evaluation_pipeline.add_metric(blip_scorer)

    diffuion_evaluation_pipeline(
        dataset="parti-prompts:v1",
        init_params=dict(project="t2i_eval", entity="geekyrakshit"),
    )
