from typing import Optional

import fire
import wandb
import weave

from hemm.eval_pipelines import StabilityAPIModel, EvaluationPipeline
from hemm.metrics.vqa import MultiModalLLMEvaluationMetric
from hemm.metrics.vqa.judges.mmllm_judges import OpenAIJudge, PromptCategory


def main(
    project="mllm-eval",
    entity="hemm-eval",
    dataset_ref: Optional[str] = "attribute_binding_dataset:v1",
    dataset_limit: Optional[int] = None,
    model_name: str = "sd3-large",
):
    wandb.init(project=project, entity=entity, job_type="evaluation")
    weave.init(project_name=f"{entity}/{project}")

    dataset = weave.ref(dataset_ref).get()
    dataset = dataset.rows[:dataset_limit] if dataset_limit else dataset

    stability_model = StabilityAPIModel(model_name=model_name)
    evaluation_pipeline = EvaluationPipeline(model=stability_model)

    judge = OpenAIJudge(prompt_property=PromptCategory.action)
    metric = MultiModalLLMEvaluationMetric(judge=judge)
    evaluation_pipeline.add_metric(metric)

    evaluation_pipeline(dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
