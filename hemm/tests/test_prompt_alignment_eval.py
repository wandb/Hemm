import asyncio

import weave

from hemm.metrics.prompt_alignment import CLIPImageQualityScoreMetric, CLIPScoreMetric
from hemm.models import DiffusersModel


def test_prompt_alignment_evaluation():
    weave.init(project_name="hemm-eval/unit-tests")
    model = DiffusersModel(
        diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
        enable_cpu_offfload=False,
    )
    clip_scorer = CLIPScoreMetric(
        clip_model_name_or_path="openai/clip-vit-base-patch16"
    )
    clip_iqa_scorer = CLIPImageQualityScoreMetric(clip_model_name_or_path="clip_iqa")
    dataset = weave.ref("parti-prompts:v0").get().rows[:2]
    evaluation = weave.Evaluation(
        dataset=dataset, scorers=[clip_scorer, clip_iqa_scorer]
    )
    asyncio.run(evaluation.evaluate(model))
