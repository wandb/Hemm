import asyncio

import fire
import weave

from hemm.metrics.prompt_alignment import CLIPImageQualityScoreMetric, CLIPScoreMetric
from hemm.models import DiffusersModel


def main(
    diffusion_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
    clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
    clip_iqa_model_name_or_path: str = "clip_iqa",
    diffusion_model_enable_cpu_offfload: bool = False,
    dataset: str = "parti-prompts:v0",
    project: str = "propmpt-alignment",
):
    weave.init(project_name=project)

    model = DiffusersModel(
        diffusion_model_name_or_path=diffusion_model_name_or_path,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
    )

    clip_scorer = CLIPScoreMetric(clip_model_name_or_path=clip_model_name_or_path)
    clip_iqa_scorer = CLIPImageQualityScoreMetric(
        clip_model_name_or_path=clip_iqa_model_name_or_path
    )

    evaluation = weave.Evaluation(
        dataset=dataset, scorers=[clip_scorer, clip_iqa_scorer]
    )
    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    fire.Fire(main)
