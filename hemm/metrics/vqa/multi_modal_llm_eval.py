from typing import Any, Dict, Optional, Union

import weave

from ..base import BaseMetric
from .judges.mmllm_judges import OpenAIJudge


class MultiModalLLMEvaluationMetric(BaseMetric):

    def __init__(
        self,
        judge: Union[weave.Model, OpenAIJudge],
        name: Optional[str] = "mmllm_eval_metric",
    ) -> None:
        super().__init__()
        self.judge = judge
        self.judge._initialize()
        self.config = self.judge.model_dump()
        self.prompt_property = judge.prompt_property
        self.scores = []
        self.name = name

    @weave.op()
    def evaluate(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, Any]:
        judgements = self.judge.predict(prompt=prompt, image=model_output["image"])
        score = sum([judgement.score for judgement in judgements]) / len(judgements)
        fractional_score = score / self.judge._total_score
        evaluation_dict = {"score": score, "fractional_score": fractional_score}
        self.scores.append(evaluation_dict)
        return evaluation_dict

    @weave.op()
    async def evaluate_async(
        self, prompt: str, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.evaluate(prompt, model_output)
