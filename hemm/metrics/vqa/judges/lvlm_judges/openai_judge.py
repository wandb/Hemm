from typing import Any, Dict, List, Optional

import spacy
import weave
from openai import OpenAI
from pydantic import BaseModel

from .....utils import str_to_json


class TaggedPromptParts(BaseModel):
    entity: str
    noun: str
    adjective: str


class OpenAIJudge(weave.Model):
    prompt_pipeline: str = "en_core_web_sm"
    object_property: str = "color"
    openai_model: str = "gpt-4o"
    system_prompt: Optional[str] = None
    _nlp_pipeline: spacy.Language = None
    _openai_client: OpenAI = None

    def _initialize(self):
        self._nlp_pipeline = spacy.load(self.prompt_pipeline)
        self._openai_client = OpenAI()
        self.system_prompt = f"""
    You are a helpful assistant meant to identify any objects and their {self.object_property} in the given image.
        """

    @weave.op()
    def extract_prompt_parts(self, prompt: str) -> List[TaggedPromptParts]:
        doc = self._nlp_pipeline(prompt)
        tagged_prompt_parts: List[TaggedPromptParts] = []
        for chunk in doc.noun_chunks:
            chunk_np = chunk.text
            for token in list(chunk)[::-1]:
                if token.pos_ == "NOUN":
                    noun = token.lemma_
                    adjective = chunk_np.replace(f" {noun}", "")
                    adjective = adjective.replace("the ", "")
                    adjective = adjective.replace("a ", "")
                    adjective = adjective.replace("an ", "")
                    break
            tagged_prompt_parts.append(
                TaggedPromptParts(entity=chunk_np, noun=noun, adjective=adjective)
            )
        return tagged_prompt_parts

    @weave.op()
    def frame_question(self, tagged_prompt_parts: List[TaggedPromptParts]) -> List[str]:
        questions: List[str] = []
        for tagged_prompt_part in tagged_prompt_parts:
            questions.append(
                f"""
            According to the image, evaluate if there is a {tagged_prompt_part.entity} in the image.
            Give a score from 0 to 100, according the criteria:
            4: there is {tagged_prompt_part.noun}, and {self.object_property} is {tagged_prompt_part.adjective}.
            3: there is {tagged_prompt_part.noun}, {self.object_property} is mostly {tagged_prompt_part.adjective}.
            2: there is {tagged_prompt_part.noun}, but it is not {tagged_prompt_part.adjective}.
            1: no {tagged_prompt_part.noun} in the image.\n\
            Provide your analysis and explanation in JSON format with the following keys: score (e.g., 1), \
            explanation (within 20 words).
                """
            )
        return questions

    @weave.op()
    def predict(self, prompt: str, image: str) -> List[Dict[str, Any]]:
        tagged_prompt_parts = self.extract_prompt_parts(prompt)
        questions = self.frame_question(tagged_prompt_parts)
        answers = []
        for question in questions:
            response = (
                self._openai_client.chat.completions.create(
                    model=self.openai_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": image}},
                            ],
                        },
                    ],
                )
                .choices[0]
                .message.content
            )
            structured_response = str_to_json(response)
            answers.append(structured_response)
        return answers
