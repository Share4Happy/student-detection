from pathlib import Path
from typing import Literal

from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

from student_detection.common import NumpyImage

from . import BoundingBox, ObjectDetectionModel


class YOLOFineTunedSingleClass(ObjectDetectionModel):
    """A YOLO object detection adapter for custom fine-tuned single-class models."""

    def __init__(
        self,
        model: str | Path,
        device: Literal["cpu", "cuda"] = "cpu",
        class_name: str | None = None,
    ):
        self.model = YOLO(model).to(device)
        self.class_name = class_name

        # Extract native training image size from model specs if available (default: 640)
        train_args = getattr(self.model, "overrides", {}) or getattr(self.model.ckpt, "train_args", {})
        self.default_imgsz: int = int(train_args.get("imgsz", 640))

    def predict(
        self,
        image: NumpyImage,
        conf_threshold: float = 0.25,
        imgsz: int | None = None,
        classes: list[int] | None = None,
        **kwargs,
    ) -> list[BoundingBox]:
        """Run inference using the fine-tuned YOLO model and return standardized BoundingBox objects."""
        inference_imgsz = imgsz if imgsz is not None else self.default_imgsz

        # If classes not explicitly specified, filter to class 0 (single-class fine-tuned target)
        filter_classes = classes if classes is not None else [0]

        # Run Ultralytics inference
        results: list[Results] = self.model.predict(
            source=image,
            conf=conf_threshold,
            classes=filter_classes,
            imgsz=inference_imgsz,
            verbose=False,
            **kwargs,
        )  # type: ignore

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
        classes_arr = r.boxes.cls.cpu().numpy().astype(int)

        # Convert proprietary YOLO results into standardized BoundingBox objects
        for box, conf, cls_id in zip(coords, confs, classes_arr):
            name = self.class_name if self.class_name is not None else self.model.names.get(cls_id, str(cls_id))
            boxes_output.append(
                BoundingBox(
                    x1=int(box[0]),
                    y1=int(box[1]),
                    x2=int(box[2]),
                    y2=int(box[3]),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=name,
                )
            )

        return boxes_output
