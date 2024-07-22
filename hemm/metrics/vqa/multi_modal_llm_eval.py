from typing import Any, Dict, Optional, Union

import weave

from ..base import BaseMetric
from .judges.mmllm_judges import OpenAIJudge, PromptCategory


class MultiModalLLMEvaluationMetric(BaseMetric):

    def __init__(
        self,
        judge: Union[weave.Model, OpenAIJudge],
        prompt_property: PromptCategory,
        name: Optional[str] = "vlm_eval_metric",
    ) -> None:
        super().__init__()
        self.judge = judge
        self.judge._initialize()
        self.config = self.judge.model_dump()
        self.prompt_property = prompt_property
        self.scores = []
        self.name = name

    @weave.op()
    def evaluate(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, Any]:
        judgements = self.judge.predict(prompt=prompt, image=model_output["image"])
        evaluation_dict = {}
        for judgement in judgements:
            evaluation_dict[judgement.question] = {
                "score": judgement.score,
                "explanation": judgement.explanation,
            }
        self.scores.append(evaluation_dict)
        return evaluation_dict

    @weave.op()
    async def evaluate_async(
        self, prompt: str, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        return super().evaluate(prompt, model_output)
