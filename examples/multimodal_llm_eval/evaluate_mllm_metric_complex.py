from typing import Optional, Tuple

import fire
import wandb
import weave

from hemm.eval_pipelines import BaseDiffusionModel, EvaluationPipeline
from hemm.metrics.vqa import MultiModalLLMEvaluationMetric
from hemm.metrics.vqa.judges.mmllm_judges import OpenAIJudge, PromptCategory


def main(
    project="mllm-eval",
    entity="hemm-eval",
    dataset_ref: Optional[str] = "attribute_binding_dataset:v1",
    dataset_limit: Optional[int] = None,
    diffusion_model_address: str = "stabilityai/stable-diffusion-2-1",
    diffusion_model_enable_cpu_offfload: bool = False,
    image_size: Tuple[int, int] = (512, 512),
):
    wandb.init(project=project, entity=entity, job_type="evaluation")
    weave.init(project_name=project)

    # dataset = weave.ref(dataset_ref).get()
    # dataset = dataset.rows[:dataset_limit] if dataset_limit else dataset
    dataset = (
        [
            {"prompt": "The black chair is on the right of the wooden table."},
            {"prompt": "The round clock was mounted on the white wall."},
            {"prompt": "The fluffy pillow was on the left of the striped blanket."},
        ]
        if dataset_limit
        else weave.ref(dataset_ref).get()
    )

    diffusion_model = BaseDiffusionModel(
        diffusion_model_name_or_path=diffusion_model_address,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
        image_height=image_size[0],
        image_width=image_size[1],
    )
    evaluation_pipeline = EvaluationPipeline(model=diffusion_model)

    judge = OpenAIJudge(prompt_property=PromptCategory.complex)
    metric = MultiModalLLMEvaluationMetric(judge=judge)
    evaluation_pipeline.add_metric(metric)

    evaluation_pipeline(dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
