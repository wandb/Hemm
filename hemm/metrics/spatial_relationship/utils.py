import cv2
import numpy as np
from PIL import Image

from .judges.commons import BoundingBox


def get_iou(entity_1: BoundingBox, entity_2: BoundingBox) -> float:
    """Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        entity_1 (BoundingBox): The first bounding box.
        entity_2 (BoundingBox): The second bounding box.

    Returns:
        float: The IoU score between the two bounding boxes.
    """
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


def annotate_with_bounding_box(image: Image.Image, entity: BoundingBox) -> Image.Image:
    """Annotate an image with a bounding box and label.

    Args:
        image (Union[str, Image.Image]): The image to annotate.
        entity (BoundingBox): The bounding box to annotate.

    Returns:
        Image.Image: The annotated image.
    """
    image = np.array(image)
    cv2.rectangle(
        image,
        (int(entity.box_coordinates_min.x), int(entity.box_coordinates_min.y)),
        (int(entity.box_coordinates_max.x), int(entity.box_coordinates_max.y)),
        color=(0, 0, 0),
        thickness=1,
    )
    cv2.putText(
        image,
        entity.label,
        (int(entity.box_coordinates_min.x), int(entity.box_coordinates_min.y) - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=2,
    )
    return Image.fromarray(image)
