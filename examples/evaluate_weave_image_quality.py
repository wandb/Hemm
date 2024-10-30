import asyncio

import fire
import weave

from hemm.metrics.image_quality import LPIPSMetric, PSNRMetric, SSIMMetric
from hemm.models import DiffusersModel


def main(
    project_name: str = "image-quality",
    diffusion_model_name_or_path="stabilityai/stable-diffusion-2-1",
    dataset_ref: str = "COCO:v0",
    image_height: int = 1024,
    image_width: int = 1024,
):
    weave.init(project_name=project_name)

    model = DiffusersModel(diffusion_model_name_or_path=diffusion_model_name_or_path)

    psnr_metric = PSNRMetric(image_size=(image_height, image_width))
    ssim_metric = SSIMMetric(image_size=(image_height, image_width))
    lpips_metric = LPIPSMetric(image_size=(image_height, image_width))

    dataset = weave.ref(dataset_ref).get()
    evaluation = weave.Evaluation(
        dataset=dataset, scorers=[psnr_metric, ssim_metric, lpips_metric]
    )
    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    fire.Fire(main)
