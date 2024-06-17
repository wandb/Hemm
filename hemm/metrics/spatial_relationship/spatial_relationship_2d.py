from typing import Any, Dict, List, Optional, Union

import weave

from .judges import DETRSpatialRelationShipJudge
from .judges.commons import BoundingBox


def get_iou(entity_1: BoundingBox, entity_2: BoundingBox) -> float:
    x_overlap = max(
        0,
        min(entity_1.box_coordinates_max.x, entity_2.box_coordinates_max.x)
        - max(entity_1.box_coordinates_min.x, entity_2.box_coordinates_min.x),
    )
    y_overlap = max(
        0,
        min(entity_1.box_coordinates_max.y, entity_2.box_coordinates_max.y)
        - max(entity_1.box_coordinates_min.y, entity_2.box_coordinates_min.y),
    )
    intersection = x_overlap * y_overlap
    box_1_area = (entity_1.box_coordinates_max.x - entity_1.box_coordinates_min.x) * (
        entity_1.box_coordinates_max.y - entity_1.box_coordinates_min.y
    )
    box_1_area = (entity_2.box_coordinates_max.x - entity_2.box_coordinates_min.x) * (
        entity_2.box_coordinates_max.y - entity_2.box_coordinates_min.y
    )
    union = box_1_area + box_1_area - intersection
    return intersection / union


class SpatialRelationshipMetric2D:
    def __init__(
        self,
        judge: Union[weave.Model, DETRSpatialRelationShipJudge],
        iou_threshold: Optional[float] = 0.1,
        distance_threshold: Optional[float] = 150,
        name: Optional[str] = "spatial_relationship_score",
    ) -> None:
        self.judge = judge
        self.judge._initialize_models()
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.name = name
        self.scores = []
        self.config = judge.model_dump()

    @weave.op()
    async def __call__(
        self, prompt: str, response: Dict[str, Any], model_output: Dict[str, Any]
    ) -> Dict[str, Union[bool, float, int]]:
        _ = prompt

        image = model_output["image"]
        boxes: List[BoundingBox] = self.judge.predict(image)

        # Determine presence of entities in the judgement
        judgement = {
            "enity_1_present": False,
            "entity_2_present": False,
        }
        entities = [entity["name"] for entity in response["entities"]]
        entity_boxes: List[BoundingBox] = [None, None]
        for box in boxes:
            if box.label == entities[0]:
                judgement["entity_1_present"] = True
                entity_boxes[0] = box
            elif box.label == entities[1]:
                judgement["entity_2_present"] = True
                entity_boxes[1] = box

        # assign score based on the spatial relationship inferred
        # from the judgement
        center_distance_x = abs(
            entity_boxes[0].box_coordinates_center.x
            - entity_boxes[1].box_coordinates_center.x
        )
        center_distance_y = abs(
            entity_boxes[0].box_coordinates_center.y
            - entity_boxes[1].box_coordinates_center.y
        )
        iou = get_iou(entity_boxes[0], entity_boxes[1])
        score = 0.0
        if response["relation"] in ["near", "next to", "on side of", "side of"]:
            if (
                abs(center_distance_x) < self.distance_threshold
                or abs(center_distance_y) < self.distance_threshold
            ):
                score = 1.0
            else:
                score = self.distance_threshold / max(
                    abs(center_distance_x), abs(center_distance_y)
                )
        elif response["relation"] == "on the right of":
            if center_distance_x < 0:
                if (
                    abs(center_distance_x) > abs(center_distance_y)
                    and iou < self.iou_threshold
                ):
                    score = 1.0
                elif (
                    abs(center_distance_x) > abs(center_distance_y)
                    and iou >= self.iou_threshold
                ):
                    score = self.iou_threshold / iou
        elif response["relation"] == "on the left of":
            if center_distance_x > 0:
                if (
                    abs(center_distance_x) > abs(center_distance_y)
                    and iou < self.iou_threshold
                ):
                    score = 1.0
                elif (
                    abs(center_distance_x) > abs(center_distance_y)
                    and iou >= self.iou_threshold
                ):
                    score = self.iou_threshold / iou
            else:
                score = 0.0
        elif response["relation"] == "on the bottom of":
            if center_distance_y < 0:
                if (
                    abs(center_distance_y) > abs(center_distance_x)
                    and iou < self.iou_threshold
                ):
                    score = 1
                elif (
                    abs(center_distance_y) > abs(center_distance_x)
                    and iou >= self.iou_threshold
                ):
                    score = self.iou_threshold / iou
        elif response["relation"] == "on the top of":
            if center_distance_y > 0:
                if (
                    abs(center_distance_y) > abs(center_distance_x)
                    and iou < self.iou_threshold
                ):
                    score = 1
                elif (
                    abs(center_distance_y) > abs(center_distance_x)
                    and iou >= self.iou_threshold
                ):
                    score = self.iou_threshold / iou
        judgement["score"] = score
        return judgement
