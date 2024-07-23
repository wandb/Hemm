from enum import Enum

from pydantic import BaseModel


class JudgeQuestion(BaseModel):
    image_desciption_system_prompt: str
    judgement_question_system_prompt: str
    judgement_question: str
    image: str


class JudgeMent(BaseModel):
    question: str
    score: int
    explanation: str


class PromptCategory(Enum):
    color = 0
    shape = 1
    texture = 2
    spatial = 3
    spatial_3d = 4
    action = 5
    numeracy = 6
    complex = 7


class TaggedPromptParts(BaseModel):
    entity: str
    noun: str
    adjective: str
