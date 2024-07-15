import os
from typing import List, Optional

import instructor
import spacy
import weave
from openai import OpenAI
from pydantic import BaseModel

from .commons import PromptCategory, TaggedPromptParts, get_system_prompt


class OpenAIJudgeMent(BaseModel):
    question: str
    score: int
    explanation: str


class OpenAIJudge(weave.Model):
    prompt_pipeline: str = "en_core_web_sm"
    prompt_property: PromptCategory = PromptCategory.color
    openai_model: str = "gpt-4o"
    system_prompt: Optional[str] = None
    _nlp_pipeline: spacy.Language = None
    _openai_client: OpenAI = None

    def _initialize(self):
        self._nlp_pipeline = spacy.load(self.prompt_pipeline)
        self._openai_client = instructor.from_openai(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        )
        self.system_prompt = get_system_prompt(self.prompt_property)

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
    def frame_question(self, prompt: str) -> List[str]:
        if self.prompt_property in [PromptCategory.spatial, PromptCategory.spatial_3d]:
            return [
                f"""
        According to the image, evaluate if the text \"{prompt}\" is correctly portrayed in the image.
        Give a score from 0 to 100, according the criteria:
        5: correct spatial layout in the image for all objects mentioned in the text.
        4: basically, spatial layout of objects matches the text.
        3: spatial layout not aligned properly with the text.
        2: image not aligned properly with the text.
        1: image almost irrelevant to the text.
        Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2),
        explanation (within 20 words).
            """
            ]
        elif self.prompt_property == PromptCategory.action:
            return [
                f"""
        According to the image, evaluate if the text \"{prompt}\" is correctly portrayed in the image.
        Give a score from 0 to 100, according the criteria:
        5: the image accurately portrayed the actions, events and relationships between objects described in the text.
        4: the image portrayed most of the actions, events and relationships but with minor discrepancies.
        3: the image depicted some elements, but action relationships between objects are not correct.
        2: the image failed to convey the full scope of the text.
        1: the image did not depict any actions or events that match the text.
        Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2),
        explanation (within 20 words).
            """
            ]
        elif self.prompt_property == PromptCategory.numeracy:
            return [
                f"""
        According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt}\"
        Give a score from 0 to 100, according the criteria:
        5: correct numerical content in the image for all objects mentioned in the text
        4: basically, numerical content of objects matches the text
        3: numerical content not aligned properly with the text
        2: image not aligned properly with the text
        1: image almost irrelevant to the text
        Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2),
        explanation (within 20 words)."
            """
            ]
        elif self.prompt_property == PromptCategory.complex:
            return [
                f"""
        According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt}\"
        Give a score from 0 to 100, according the criteria:
        5: the image perfectly matches the content of the text prompt, with no discrepancies.
        4: the image portrayed most of the actions, events and relationships but with minor discrepancies.
        3: the image depicted some elements in the text prompt, but ignored some key parts or details.
        2: the image did not depict any actions or events that match the text.
        1: the image failed to convey the full scope in the text prompt.
        Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2),
        explanation (within 20 words).
            """
            ]
        tagged_prompt_parts = self.extract_prompt_parts(prompt)
        questions: List[str] = []
        for tagged_prompt_part in tagged_prompt_parts:
            questions.append(
                f"""
            According to the image, evaluate if there is a {tagged_prompt_part.entity} in the image.
            Give a score from 0 to 100, according the criteria:
            4: there is {tagged_prompt_part.noun}, and {self.prompt_property.name} is {tagged_prompt_part.adjective}.
            3: there is {tagged_prompt_part.noun}, {self.prompt_property.name} is mostly {tagged_prompt_part.adjective}.
            2: there is {tagged_prompt_part.noun}, but it is not {tagged_prompt_part.adjective}.
            1: no {tagged_prompt_part.noun} in the image.\n\
            Provide your analysis and explanation in JSON format with the following keys: score (e.g., 1), \
            explanation (within 20 words).
                """
            )
        return questions

    @weave.op()
    def predict(self, prompt: str, image: str) -> List[OpenAIJudgeMent]:
        questions = self.frame_question(prompt)
        answers = []
        for question in questions:
            response = (
                self._openai_client.chat.completions.create(
                    model=self.openai_model,
                    response_model=OpenAIJudgeMent,
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
            answers.append(response)
        return answers
