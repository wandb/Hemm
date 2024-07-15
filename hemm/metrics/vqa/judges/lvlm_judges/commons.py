from enum import Enum

import weave
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


@weave.op()
def get_system_prompt(prompt_property: PromptCategory) -> str:
    if prompt_property in [PromptCategory.spatial, PromptCategory.spatial_3d]:
        return """
    You are a helpful assistant meant to identify objects and their spatial layout in the image.
    You have to extract the score and the explanation from the user's response.
        """
    elif prompt_property == PromptCategory.action:
        return """
    You are a helpful assistant meant to identify the actions, events, objects and their relationships in the image.
    You have to extract the score and the explanation from the user's response.
        """
    elif prompt_property == PromptCategory.numeracy:
        return """
    You are a helpful assistant meant to identify objects and their quantities in the image.
    You have to extract the score and the explanation from the user's response.
        """
    elif prompt_property == PromptCategory.complex:
        return """
    You are a helpful assistant meant to evaluate the correspondence of the image to a given text prompt.
    focus on the objects in the image and their attributes (such as color, shape, texture),
    spatial layout and action relationships. You have to extract the score and the explanation from the
    user's response.
        """
    return f"""
    You are a helpful assistant meant to identify any objects and their {prompt_property.name}
    in the given image. You have to extract the score and the explanation from the user's response.
        """
