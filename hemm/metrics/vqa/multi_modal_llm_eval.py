from typing import Any, Dict, List, Optional, Union

import weave

from ..base import BaseMetric
from .judges.mmllm_judges import OpenAIJudge
from .judges.mmllm_judges.openai_judge import OpenAIJudgeMent


class MultiModalLLMEvaluationMetric(BaseMetric):
    """Multi-modal LLM-based evaluation metric for an image-generation model.

    Args:
        judge (Union[weave.Model, OpenAIJudge]): The judge LLM model to evaluate the generated images.
        name (Optional[str]): Name of the evaluation.
    """

    def __init__(
        self,
        judge: Union[weave.Model, OpenAIJudge],
        name: Optional[str] = "mmllm_eval_metric",
    ) -> None:
        super().__init__()
        self.judge = judge
        self.config = self.judge.model_dump()
        self.prompt_property = judge.prompt_property
        self.scores = []
        self.name = name

    @weave.op()
    def evaluate(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the generated image using the judge LLM model.

        Args:
            prompt (str): The prompt for the model.
            model_output (Dict[str, Any]): The model output.
        """
        judgements: List[OpenAIJudgeMent] = self.judge.predict(
            prompt=prompt, image=model_output["image"]
        )
        score = sum([judgement.judgement.score for judgement in judgements])
        fractional_score = sum([judgement.fractional_score for judgement in judgements])
        evaluation_dict = {
            "score": score / len(judgements),
            "fractional_score": fractional_score / len(judgements),
        }
        self.scores.append(evaluation_dict)
        return evaluation_dict

    @weave.op()
    async def evaluate_async(
        self, prompt: str, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.evaluate(prompt, model_output)
