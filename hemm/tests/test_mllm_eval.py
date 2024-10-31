import asyncio

import weave

from hemm.metrics.vqa import MultiModalLLMEvaluationMetric
from hemm.metrics.vqa.judges.mmllm_judges import OpenAIJudge, PromptCategory
from hemm.models import DiffusersModel


def test_multimodal_llm_evaluation():
    weave.init(project_name="hemm-eval/unit-tests")
    model = DiffusersModel(
        diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4",
        enable_cpu_offfload=False,
        image_height=1024,
        image_width=1024,
    )

    judge = OpenAIJudge(prompt_property=PromptCategory.complex)
    metric = MultiModalLLMEvaluationMetric(judge=judge)

    dataset = [
        {"prompt": "The fluffy pillow was on the left of the striped blanket."},
        {"prompt": "The round clock was mounted on the white wall."},
        {"prompt": "The black chair is on the right of the wooden table."},
    ]

    evaluation = weave.Evaluation(dataset=dataset, scorers=[metric])
    asyncio.run(evaluation.evaluate(model))
