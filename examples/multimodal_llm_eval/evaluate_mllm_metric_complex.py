from typing import Optional

import fire
import weave

import wandb
from hemm.eval_pipelines import EvaluationPipeline
from hemm.metrics.vqa import MultiModalLLMEvaluationMetric
from hemm.metrics.vqa.judges.mmllm_judges import OpenAIJudge, PromptCategory
from hemm.models import BaseDiffusionModel


def main(
    project="mllm-eval",
    entity="hemm-eval",
    dataset_ref: Optional[str] = "attribute_binding_dataset:v1",
    dataset_limit: Optional[int] = None,
    diffusion_model_address: str = "stabilityai/stable-diffusion-2-1",
    diffusion_model_enable_cpu_offfload: bool = False,
    openai_judge_model: str = "gpt-4o",
    image_height: int = 1024,
    image_width: int = 1024,
    num_inference_steps: int = 50,
):
    wandb.init(project=project, entity=entity, job_type="evaluation")
    weave.init(project_name=f"{entity}/{project}")

    dataset = weave.ref(dataset_ref).get()
    dataset = dataset.rows[:dataset_limit] if dataset_limit else dataset

    diffusion_model = BaseDiffusionModel(
        diffusion_model_name_or_path=diffusion_model_address,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
        image_height=image_height,
        image_width=image_width,
        num_inference_steps=num_inference_steps,
    )
    diffusion_model._pipeline.set_progress_bar_config(disable=True)
    evaluation_pipeline = EvaluationPipeline(model=diffusion_model)

    judge = OpenAIJudge(
        prompt_property=PromptCategory.action, openai_model=openai_judge_model
    )
    metric = MultiModalLLMEvaluationMetric(judge=judge)
    evaluation_pipeline.add_metric(metric)

    evaluation_pipeline(dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
