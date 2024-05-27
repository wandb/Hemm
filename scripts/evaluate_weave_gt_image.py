from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.image_quality import PSNRMetric, SSIMMetric


if __name__ == "__main__":
    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        "CompVis/stable-diffusion-v1-4"
    )

    # Add PSNR Metric
    psnr_metric = PSNRMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(psnr_metric)

    # Add SSIM Metric
    ssim_metric = SSIMMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(ssim_metric)

    diffuion_evaluation_pipeline(
        dataset="COCO:v1",
        init_params=dict(project="t2i_eval", entity="geekyrakshit"),
    )
