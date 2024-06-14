from typing import Any, Dict, List, Union

import weave

from .judges import DETRSpatialRelationShipJudge
from .judges.commons import BoundingBox


class SpatialRelationshipMetric:
    def __init__(
        self, judge: Union[weave.Model, DETRSpatialRelationShipJudge], name: str
    ) -> None:
        self.judge = judge
        self.name = name
        self.scores = []
        self.config = judge.model_dump()

    @weave.op()
    async def __call__(
        self, response: Dict[str, Any], model_output: Dict[str, Any]
    ) -> Dict[str, Union[bool, float, int]]:
        image = model_output["image"][0]
        boxes: List[BoundingBox] = self.judge.predict(image)
        judgement = {
            "enity_1_present": False,
            "entity_2_present": False,
        }
        entities = [entity["name"] for entity in response["entities"]]
        for box in boxes:
            if box.label == entities[0]:
                judgement["enity_1_present"] = True
            elif box.label == entities[1]:
                judgement["entity_2_present"] = True
        return judgement
