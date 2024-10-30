import asyncio

import weave

from hemm.metrics.image_quality import LPIPSMetric, PSNRMetric, SSIMMetric
from hemm.models import DiffusersModel


def test_image_quality_metrics():
    weave.init(project_name="hemm-eval/unit-tests")
    model = DiffusersModel(
        diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
        enable_cpu_offfload=False,
    )
    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric(
        image_height=model.image_height, image_width=model.image_width
    )
    lpips_metric = LPIPSMetric(
        image_height=model.image_height, image_width=model.image_width
    )
    dataset = weave.ref("COCO:v1").get().rows[:2]
    evaluation = weave.Evaluation(
        dataset=dataset, scorers=[psnr_metric, ssim_metric, lpips_metric]
    )
    asyncio.run(evaluation.evaluate(model))
