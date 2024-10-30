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
    psnr_metric = PSNRMetric(image_size=(model.image_height, model.image_width))
    ssim_metric = SSIMMetric(image_size=(model.image_height, model.image_width))
    lpips_metric = LPIPSMetric(image_size=(model.image_height, model.image_width))
    dataset = weave.ref("COCO:v1").get().rows[:2]
    evaluation = weave.Evaluation(
        dataset=dataset, scorers=[psnr_metric, ssim_metric, lpips_metric]
    )
    summary = asyncio.run(evaluation.evaluate(model))

    assert summary["PSNRMetric.evaluate_async"]["peak_signal_noise_ratio"]["mean"] > 0
    assert (
        summary["SSIMMetric.evaluate_async"]["structural_similarity_index_measure"][
            "mean"
        ]
        > 0
    )
    assert (
        summary["LPIPSMetric.evaluate_async"][
            "alexnet_learned_perceptual_image_patch_similarity"
        ]["mean"]
        > 0.0
    )
    assert summary["model_latency"]["mean"] > 0.0
