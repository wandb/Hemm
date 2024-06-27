from typing import Any, Dict, Optional, Union

import weave
from ..base import BaseMetric
from .judges import BlipVQAJudge


class DisentangledVQAMetric(BaseMetric):

    def __init__(
        self,
        judge: Union[weave.Model, BlipVQAJudge],
        name: Optional[str] = "disentangled_vlm_metric",
    ) -> None:
        super().__init__()
        self.judge = judge
        self.judge._initialize_models()
        self.config = self.judge.model_dump()
        self.scores = []
        self.name = name

    @weave.op()
    def evaluate(
        self,
        prompt: str,
        adj_1: str,
        noun_1: str,
        adj_2: str,
        noun_2: str,
        model_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = prompt
        judgement = self.judge.predict(
            adj_1, noun_1, adj_2, noun_2, model_output["image"]
        )
        self.scores.append(judgement)
        return judgement

    @weave.op()
    async def evaluate(
        self,
        prompt: str,
        adj_1: str,
        noun_1: str,
        adj_2: str,
        noun_2: str,
        model_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        return super().evaluate(prompt, adj_1, noun_1, adj_2, noun_2, model_output)
