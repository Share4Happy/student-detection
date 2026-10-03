from pathlib import Path
from typing import Literal

from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

from student_detection.common import NumpyImage

from . import BoundingBox, ObjectDetectionModel


class YOLOPerson(ObjectDetectionModel):
    """A YOLO object detection adapter that only detects person"""
    def __init__(
        self,
        model: str | Path = "yolo26n.pt",
        device: Literal["cpu", "cuda"] = "cpu"
    ):
        self.model = YOLO(model).to(device)

    def predict(
        self,
        image: NumpyImage,
        conf_threshold: float = 0.25,
        imgsz: int = 1280,
        **kwargs,
    ) -> list[BoundingBox]:
        """
        Run inference using the YOLO model and return standardized BoundingBox of `person` class.
        """

        # Run Ultralytics inference
        results: list[Results] = self.model.predict(
            source=image,
            conf=conf_threshold,
            classes=[0], # 0 is id for "person" class
            imgsz=imgsz,
            verbose=False,
            **kwargs,
        ) # type: ignore

        boxes_output: list[BoundingBox] = []

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return boxes_output

        # Extract coordinates, confidences, and class IDs as CPU numpy arrays
        assert isinstance(r.boxes.xyxy, Tensor)
        assert isinstance(r.boxes.conf, Tensor)
        assert isinstance(r.boxes.cls, Tensor)

        coords = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Convert proprietary YOLO results into standardized BoundingBox objects
        for box, conf, cls_id in zip(coords, confs, classes):
            class_name = self.model.names.get(cls_id, str(cls_id))
            boxes_output.append(
                BoundingBox(
                    x1=int(box[0]),
                    y1=int(box[1]),
                    x2=int(box[2]),
                    y2=int(box[3]),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                )
            )

        return boxes_output
