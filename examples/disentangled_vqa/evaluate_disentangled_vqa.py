import asyncio
from typing import Optional

import fire
import weave

from hemm.metrics.vqa import DisentangledVQAMetric
from hemm.metrics.vqa.judges import BlipVQAJudge
from hemm.models import DiffusersModel


def main(
    project="disentangled_vqa",
    entity="hemm-eval",
    dataset_ref: Optional[str] = "attribute_binding_dataset:v1",
    dataset_limit: Optional[int] = None,
    diffusion_model_address: str = "stabilityai/stable-diffusion-2-1",
    diffusion_model_enable_cpu_offfload: bool = False,
    image_height: int = 1024,
    image_width: int = 1024,
):
    weave.init(project_name=f"{entity}/{project}")

    dataset = weave.ref(dataset_ref).get()
    dataset = dataset.rows[:dataset_limit] if dataset_limit else dataset

    model = DiffusersModel(
        diffusion_model_name_or_path=diffusion_model_address,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
        image_height=image_height,
        image_width=image_width,
    )
    model._pipeline.set_progress_bar_config(disable=True)

    judge = BlipVQAJudge()
    metric = DisentangledVQAMetric(judge=judge, name="disentangled_blip_metric")

    evaluation = weave.Evaluation(dataset=dataset, scorers=[metric])
    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    fire.Fire(main)
