import base64
from io import BytesIO
from typing import List

import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

import weave

from .commons import BoundingBox, CartesianCoordinate2D
from ....utils import base64_decode_image


class DETRSpatialRelationShipJudge(weave.Model):
    """DETR spatial relationship judge model for 2D images.

    Args:
        model_address (str, optional): The address of the model to use.
        revision (str, optional): The revision of the model to use.
    """

    model_address: str = "facebook/detr-resnet-50"
    revision: str = "no_timm"
    _feature_extractor: DetrImageProcessor = None
    _object_detection_model: DetrForObjectDetection = None

    def _initialize_models(self):
        self._feature_extractor = DetrImageProcessor.from_pretrained(
            self.model_address, revision=self.revision
        )
        self._object_detection_model = DetrForObjectDetection.from_pretrained(
            self.model_address, revision=self.revision
        )

    @weave.op()
    def predict(self, image: str) -> List[BoundingBox]:
        """Predict the bounding boxes from the input image.

        Args:
            image (str): The base64 encoded image.

        Returns:
            List[BoundingBox]: The predicted bounding boxes.
        """
        pil_image = base64_decode_image(image)
        encoding = self._feature_extractor(pil_image, return_tensors="pt")
        outputs = self._object_detection_model(**encoding)
        target_sizes = torch.tensor([pil_image.size[::-1]])
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
