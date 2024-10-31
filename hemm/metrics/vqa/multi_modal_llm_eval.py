from typing import Any, Dict, List

import weave

from .judges.mmllm_judges import OpenAIJudge
from .judges.mmllm_judges.openai_judge import OpenAIJudgeMent


class MultiModalLLMEvaluationMetric(weave.Scorer):
    """Multi-modal LLM-based evaluation metric for an image-generation model.

    Args:
        judge (OpenAIJudge): The judge LLM model to evaluate the generated images.
    """

    judge: OpenAIJudge

    @weave.op()
    def score(self, prompt: str, model_output: Dict[str, Any]) -> Dict[str, Any]:
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
        return evaluation_dict
