import asyncio

import weave

from hemm.metrics.vqa import DisentangledVQAMetric
from hemm.metrics.vqa.judges import BlipVQAJudge
from hemm.models import DiffusersModel


def test_disentangled_vqa_evaluation():
    weave.init(project_name="hemm-eval/unit-tests")
    model = DiffusersModel(
        diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
        enable_cpu_offfload=False,
    )

    judge = BlipVQAJudge()
    metric = DisentangledVQAMetric(judge=judge, name="disentangled_blip_metric")

    dataset = weave.ref("attribute_binding_dataset:v0").get().rows[:2]
    evaluation = weave.Evaluation(dataset=dataset, scorers=[metric])
    asyncio.run(evaluation.evaluate(model))
