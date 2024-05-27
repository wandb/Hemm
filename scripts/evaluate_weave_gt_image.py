from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.image_quality import PSNRMetric


if __name__ == "__main__":
    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        "CompVis/stable-diffusion-v1-4"
    )

    # Add FID Metric
    fid_metric = PSNRMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(fid_metric)

    diffuion_evaluation_pipeline(
        dataset="COCO:v1",
        init_params=dict(project="t2i_eval", entity="geekyrakshit"),
    )
