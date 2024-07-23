import fire
import wandb
import weave

from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.image_quality import LPIPSMetric, PSNRMetric, SSIMMetric


def main(
    project_name: str = "image-quality",
    diffusion_model_name_or_path="stabilityai/stable-diffusion-2-1",
    dataset_ref: str = "COCO:v0",
):
    wandb.init(project=project_name, job_type="evaluation")
    weave.init(project_name=project_name)

    model = BaseDiffusionModel(
        diffusion_model_name_or_path=diffusion_model_name_or_path
    )
    evaluation_pipeline = EvaluationPipeline(model=model)

    # Add PSNR Metric
    psnr_metric = PSNRMetric(image_size=evaluation_pipeline.image_size)
    evaluation_pipeline.add_metric(psnr_metric)

    # Add SSIM Metric
    ssim_metric = SSIMMetric(image_size=evaluation_pipeline.image_size)
    evaluation_pipeline.add_metric(ssim_metric)

    # Add LPIPS Metric
    lpips_metric = LPIPSMetric(image_size=evaluation_pipeline.image_size)
    evaluation_pipeline.add_metric(lpips_metric)

    evaluation_pipeline(dataset=dataset_ref)


if __name__ == "__main__":
    fire.Fire(main)
