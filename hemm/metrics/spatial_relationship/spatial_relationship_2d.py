from typing import Any, Dict, List, Optional, Union

import wandb
import weave
from PIL import Image

from .judges import DETRSpatialRelationShipJudge
from .judges.commons import BoundingBox
from .utils import annotate_with_bounding_box, get_iou
from ...utils import base64_decode_image, base64_encode_image


class SpatialRelationshipMetric2D:
    """Spatial relationship metric for 2D images as proposed by Section 4.2 from the paper
    [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/pdf/2307.06350).

    Args:
        judge (Union[weave.Model, DETRSpatialRelationShipJudge]): The judge model to predict
            the bounding boxes from the generated image.
        iou_threshold (Optional[float], optional): The IoU threshold for the spatial relationship.
        distance_threshold (Optional[float], optional): The distance threshold for the spatial relationship.
        name (Optional[str], optional): The name of the metric.
    """

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
    def compose_judgement(
        self,
        image: str,
        entity_1: str,
        entity_2: str,
        relationship: str,
        boxes: List[BoundingBox],
    ) -> Dict[str, Any]:
        """Compose the judgement based on the response and the predicted bounding boxes.

        Args:
            image (str): The base64 encoded image.
            response (Dict[str, Any]): The response from the model.
            boxes (List[BoundingBox]): The predicted bounding boxes.

        Returns:
            Dict[str, Any]: The comprehensive spatial relationship judgement.
        """
        # Determine presence of entities in the judgement
        judgement = {
            "entity_1_present": False,
            "entity_2_present": False,
        }
        entity_1_box: BoundingBox = None
        entity_2_box: BoundingBox = None
        annotated_image = image
        for box in boxes:
            if box.label == entity_1:
                judgement["entity_1_present"] = True
                entity_1_box = box
            elif box.label == entity_2:
                judgement["entity_2_present"] = True
                entity_2_box = box
            annotated_image = annotate_with_bounding_box(annotated_image, box)

        judgement["score"] = 0.0
        # assign score based on the spatial relationship inferred from the judgement
        if judgement["entity_1_present"] and judgement["entity_2_present"]:
            center_distance_x = abs(
                entity_1_box.box_coordinates_center.x
                - entity_2_box.box_coordinates_center.x
            )
            center_distance_y = abs(
                entity_1_box.box_coordinates_center.y
                - entity_2_box.box_coordinates_center.y
            )
            iou = get_iou(entity_1_box, entity_2_box)
            score = 0.0
            if relationship in ["near", "next to", "on side of", "side of"]:
                if (
                    abs(center_distance_x) < self.distance_threshold
                    or abs(center_distance_y) < self.distance_threshold
                ):
                    score = 1.0
                else:
                    score = self.distance_threshold / max(
                        abs(center_distance_x), abs(center_distance_y)
                    )
            elif relationship == "on the right of":
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
            elif relationship == "on the left of":
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
            elif relationship == "on the bottom of":
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
            elif relationship == "on the top of":
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

        self.scores.append(
            {
                **judgement,
                **{
                    "judge_annotated_image": wandb.Image(
                        base64_decode_image(annotated_image)
                        if isinstance(annotated_image, str)
                        else annotated_image
                    )
                },
            }
        )
        return {
            **judgement,
            **{
                "judge_annotated_image": (
                    base64_encode_image(annotated_image)
                    if isinstance(annotated_image, Image.Image)
                    else annotated_image
                )
            },
        }

    @weave.op()
    async def __call__(
        self,
        prompt: str,
        entity_1: str,
        entity_2: str,
        relationship: str,
        model_output: Dict[str, Any],
    ) -> Dict[str, Union[bool, float, int]]:
        """Calculate the spatial relationship score for the given prompt and model output.

        Args:
            prompt (str): The prompt for the model.
            entity_1 (str): The first entity in the spatial relationship.
            entity_2 (str): The second entity in the spatial relationship.
            relationship (str): The spatial relationship between the two entities.
            model_output (Dict[str, Any]): The output from the model.

        Returns:
            Dict[str, Union[bool, float, int]]: The comprehensive spatial relationship judgement.
        """
        _ = prompt

        image = model_output["image"]
        boxes: List[BoundingBox] = self.judge.predict(image)
        judgement = self.compose_judgement(
            image, entity_1, entity_2, relationship, boxes
        )
        return judgement
