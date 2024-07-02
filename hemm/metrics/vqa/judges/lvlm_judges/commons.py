from enum import Enum
from pydantic import BaseModel


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
