import unittest

import weave

import wandb
from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.image_quality import LPIPSMetric, PSNRMetric, SSIMMetric


class TestImageQualityEvaluation(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        wandb.init(
            project="unit-tests",
            entity="hemm-eval",
            job_type="test_image_quality_evaluation",
        )
        weave.init(project_name="hemm-eval/unit-tests")

    def test_image_quality_metrics(self):
        model = BaseDiffusionModel(
            diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
            enable_cpu_offfload=False,
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

        dataset = weave.ref("COCO:v1").get().rows[:2]
        summary = evaluation_pipeline(dataset=dataset)

        self.assertGreater(
            summary["PSNRMetric.evaluate_async"]["peak_signal_noise_ratio"]["mean"], 0.0
        )
        self.assertGreater(
            summary["SSIMMetric.evaluate_async"]["structural_similarity_index_measure"][
                "mean"
            ],
            0.0,
        )
        self.assertGreater(
            summary["LPIPSMetric.evaluate_async"][
                "alexnet_learned_perceptual_image_patch_similarity"
            ]["mean"],
            0.0,
        )
        self.assertGreater(summary["model_latency"]["mean"], 0.0)
