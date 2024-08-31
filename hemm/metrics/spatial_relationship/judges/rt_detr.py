from typing import List

import torch
import weave
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from .commons import BoundingBox, CartesianCoordinate2D


class RTDETRSpatialRelationShipJudge(weave.Model):
    """[RT-DETR](https://huggingface.co/docs/transformers/en/model_doc/rt_detr) spatial relationship judge model for 2D images.

    Args:
        model_address (str, optional): The address of the model to use.
        revision (str, optional): The revision of the model to use.
        name (str, optional): The name of the judge model
    """

    model_address: str
    name: str
    _feature_extractor: RTDetrImageProcessor = None
    _object_detection_model: RTDetrForObjectDetection = None

    def __init__(
        self,
        model_address: str = "facebook/detr-resnet-50",
        name: str = "detr_spatial_relationship_judge",
    ):
        super().__init__(model_address=model_address, name=name)
        self._feature_extractor = RTDetrImageProcessor.from_pretrained(
            self.model_address
        )
        self._object_detection_model = RTDetrForObjectDetection.from_pretrained(
            self.model_address
        )

    @weave.op()
    def predict(self, image: Image.Image) -> List[BoundingBox]:
        """Predict the bounding boxes from the input image.

        Args:
            image (Image.Image): The input image.

        Returns:
            List[BoundingBox]: The predicted bounding boxes.
        """
        encoding = self._feature_extractor(image, return_tensors="pt")
        outputs = self._object_detection_model(**encoding)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._feature_extractor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]
        bboxes = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            xmin, ymin, xmax, ymax = box.tolist()
            bboxes.append(
                BoundingBox(
                    box_coordinates_min=CartesianCoordinate2D(x=xmin, y=ymin),
                    box_coordinates_max=CartesianCoordinate2D(x=xmax, y=ymax),
                    box_coordinates_center=CartesianCoordinate2D(
                        x=(xmin + xmax) / 2, y=(ymin + ymax) / 2
                    ),
                    label=self._object_detection_model.config.id2label[label.item()],
                    score=score.item(),
                )
            )
        return bboxes
