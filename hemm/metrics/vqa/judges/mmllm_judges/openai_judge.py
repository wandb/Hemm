import os
from typing import List, Optional

import instructor
import spacy
import weave
from openai import OpenAI
from pydantic import BaseModel

from .commons import (
    PromptCategory,
    TaggedPromptParts,
    JudgeMent,
    JudgeQuestion,
)


class OpenAIJudgeMent(BaseModel):
    judgement: JudgeMent
    fractional_score: float


class OpenAIJudge(weave.Model):
    """OpenAI judge model for evaluating the generated images. The model uses
    OpenAI's GPT-4 model to evaluate the alignment of the generated images to
    the respective prompts using a chain-of-thought prompting strategy. The model
    is inspired by Section IV.D of the paper
    [T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation](https://karine-h.github.io/T2I-CompBench-new/)
    and Section 4.4 of the paper
    [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350).
    
    Args:
        prompt_pipeline (str): The Spacy pipeline to use for extracting the prompt parts.
        prompt_property (PromptCategory): The property of the prompt to evaluate.
        openai_model (str): The OpenAI model to use for evaluation.
        max_retries (int): The maximum number of retries for the OpenAI model.
        seed (int): Seed value for the random number generator.
        system_prompt (Optional[str]): The system prompt for the OpenAI model
    """
    prompt_pipeline: str = "en_core_web_sm"
    prompt_property: PromptCategory = PromptCategory.color
    openai_model: str = "gpt-4-turbo"
    max_retries: int = 5
    seed: int = 42
    _nlp_pipeline: spacy.Language = None
    _openai_client: OpenAI = None
    _instructor_openai_client: instructor.Instructor = None
    _total_score: int = 4

    def _initialize(self):
        self._nlp_pipeline = spacy.load(self.prompt_pipeline)
        self._openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._instructor_openai_client = instructor.from_openai(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        )

    @weave.op()
    def extract_prompt_parts(self, prompt: str) -> List[TaggedPromptParts]:
        """Extract the prompt parts from the given prompt.
        
        Args:
            prompt (str): The prompt to extract the parts from.
        
        Returns:
            List[TaggedPromptParts]: List of tagged prompt objects.
        """
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
    def frame_question(self, prompt: str, image: str) -> List[JudgeQuestion]:
        """Frame the question corresponding to the given prompt and image for
        the chain-of-thought system of judgement.
        
        Args:
            prompt (str): The prompt to frame the question for.
            image (str): The image to frame the question for.
        
        Returns:
            List[JudgeQuestion]: List of questions to ask for the given prompt.
        """
        if self.prompt_property in [PromptCategory.spatial, PromptCategory.spatial_3d]:
            self._total_score = 5
            question = JudgeQuestion(
                image_desciption_system_prompt="""
You are a helpful assistant meant to describe images is detail.
You should pay special attention to the objects and their spatial layout in the image.
                """,
                judgement_question_system_prompt="""
You are a helpful assistant meant to identify objects and their spatial layout in the image.
You have to extract the question, the score, and the explanation from the user's response.
                """,
                judgement_question=f"""
Looking at the image and given a detailed description of the image, evaluate if the text \"{prompt}\" is correctly portrayed in the image.
Give a score from 1 to 5, according to the following criteria:

5: correct spatial layout in the image for all objects mentioned in the text.
4: basically, spatial layout of objects matches the text.
3: spatial layout not aligned properly with the text.
2: image not aligned properly with the text.
1: image almost irrelevant to the text.
                """,
                image=image,
            )
            return [question]
        elif self.prompt_property == PromptCategory.action:
            self._total_score = 5
            question = JudgeQuestion(
                image_desciption_system_prompt="""
You are a helpful assistant meant to describe images is detail.
You should pay special attention to the the actions, events, objects and their relationships in the image.
                """,
                judgement_question_system_prompt="""
You are a helpful assistant meant to identify the actions, events, objects and their relationships in the image.
You have to extract the question, the score, and the explanation from the user's response.
                """,
                judgement_question=f"""
Looking at the image and given a detailed description of the image, evaluate if the text \"{prompt}\" is correctly portrayed in the image.
Give a score from 1 to 5, according to the following criteria:

5: the image accurately portrayed the actions, events and relationships between objects described in the text.
4: the image portrayed most of the actions, events and relationships but with minor discrepancies.
3: the image depicted some elements, but action relationships between objects are not correct.
2: the image failed to convey the full scope of the text.
1: the image did not depict any actions or events that match the text.
                """,
                image=image,
            )
            return [question]
        elif self.prompt_property == PromptCategory.numeracy:
            self._total_score = 5
            question = JudgeQuestion(
                image_desciption_system_prompt="""
You are a helpful assistant meant to describe images is detail.
You should pay special attention to the objects and their quantities in the image.
                """,
                judgement_question_system_prompt="""
You are a helpful assistant meant to identify objects and their quantities in the image.
You have to extract the question, the score, and the explanation from the user's response.
                """,
                judgement_question=f"""
Looking at the image and given a detailed description of the image, evaluate how well the image aligns with the text prompt: \"{prompt}\"
Give a score from 1 to 5, according to the following criteria:

5: correct numerical content in the image for all objects mentioned in the text
4: basically, numerical content of objects matches the text
3: numerical content not aligned properly with the text
2: image not aligned properly with the text
1: image almost irrelevant to the text
                """,
                image=image,
            )
            return [question]
        elif self.prompt_property == PromptCategory.complex:
            self._total_score = 5
            question = JudgeQuestion(
                image_desciption_system_prompt="""
You are a helpful assistant meant to describe images is detail.
You should pay special attention to the objects in the image and their attributes
(such as color, shape, texture), spatial layout and action relationships.
                """,
                judgement_question_system_prompt="""
You are a helpful assistant meant to evaluate the correspondence of the image to a given text prompt.
Focus on the objects in the image and their attributes (such as color, shape, texture),
spatial layout and action relationships. You have to extract the question, the score, and the
explanation from the user's response.
                """,
                judgement_question=f"""
Looking at the image and given a detailed description of the image, evaluate how well the image aligns with the text prompt: \"{prompt}\"
Give a score from 1 to 5, according to the following criteria:

5: the image perfectly matches the content of the text prompt with no discrepancies.
4: the image portrayed most of the actions, events and relationships but with minor discrepancies.
3: the image depicted some elements in the text prompt, but ignored some key parts or details.
2: the image did not depict any actions or events that match the text.
1: the image failed to convey the full scope in the text prompt.
                """,
                image=image,
            )
            return [question]
        tagged_prompt_parts = self.extract_prompt_parts(prompt)
        questions: List[str] = []
        for tagged_prompt_part in tagged_prompt_parts:
            question = JudgeQuestion(
                image_desciption_system_prompt=f"""
You are a helpful assistant meant to describe images is detail.
You should pay special attention to any objects and their {self.prompt_property.name} in the given image.
                """,
                judgement_question_system_prompt=f"""
You are a helpful assistant meant to identify any objects and their {self.prompt_property.name}
in the given image. You have to extract the question, the score, and the explanation from the user's response.
                """,
                judgement_question=f"""
Looking at the image and given a detailed description of the image, evaluate if there is a {tagged_prompt_part.entity} in the image.
Give a score from 1 to 4, according to the following criteria:

4: there is {tagged_prompt_part.noun}, and {self.prompt_property.name} is {tagged_prompt_part.adjective}.
3: there is {tagged_prompt_part.noun}, {self.prompt_property.name} is mostly {tagged_prompt_part.adjective}.
2: there is {tagged_prompt_part.noun}, but it is not {tagged_prompt_part.adjective}.
1: no {tagged_prompt_part.noun} in the image.
                """,
                image=image,
            )
            questions.append(question)
        return questions

    @weave.op
    def execute_chain_of_thought(self, question: JudgeQuestion) -> OpenAIJudgeMent:
        image_description_explanation = (
            self._openai_client.chat.completions.create(
                model=self.openai_model,
                seed=self.seed,
                messages=[
                    {
                        "role": "system",
                        "content": question.image_desciption_system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": question.image}},
                        ],
                    },
                ],
            )
            .choices[0]
            .message.content
        )
        question.judgement_question += f"""

Here is a detailed explanation of the image:
---
{image_description_explanation}
---

Provide your analysis and explanation to justify the score.
        """
        judgement_response = self._instructor_openai_client.chat.completions.create(
            model=self.openai_model,
            response_model=JudgeMent,
            max_retries=self.max_retries,
            seed=self.seed,
            messages=[
                {
                    "role": "system",
                    "content": question.judgement_question_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question.judgement_question},
                        {"type": "image_url", "image_url": {"url": question.image}},
                    ],
                },
            ],
        )
        return judgement_response

    @weave.op()
    def predict(self, prompt: str, image: str) -> List[OpenAIJudgeMent]:
        """Predict the score for the given prompt and image.
        
        Args:
            prompt (str): The prompt to evaluate.
            image (str): The image to evaluate.
        """
        questions = self.frame_question(prompt, image)
        answers = []
        for question in questions:
            judgement_response: JudgeMent = self.execute_chain_of_thought(question)
            judgement_response.explanation = (
                f"The score is {judgement_response.score}/{self._total_score}. "
                + judgement_response.explanation
            )
            answers.append(
                OpenAIJudgeMent(
                    judgement=judgement_response,
                    fractional_score=judgement_response.score / self._total_score,
                )
            )
        return answers
