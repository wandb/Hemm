from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.image_quality import FIDScore, LPIPSMetric, PSNRMetric, SSIMMetric


if __name__ == "__main__":
    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        "stabilityai/stable-diffusion-2-1"
    )

    # Add FID Metric
    fid_metric = FIDScore(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(fid_metric)

    # Add PSNR Metric
    psnr_metric = PSNRMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(psnr_metric)

    # Add SSIM Metric
    ssim_metric = SSIMMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(ssim_metric)

    # Add LPIPS Metric
    lpips_metric = LPIPSMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(lpips_metric)

    diffuion_evaluation_pipeline(
        dataset="COCO:v0",
        init_params=dict(project="t2i_eval", entity="geekyrakshit"),
    )
