from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from student_detection.common import NumpyImage

from . import SegmentationResult, SemanticSegmentationModel


class YOLOSemanticPerson(SemanticSegmentationModel):
    """A YOLO semantic segmentation adapter that only segments person"""
    def __init__(
        self,
        model: str | Path = "yolo26n-sem.pt",
        device: Literal["cpu", "cuda"] = "cpu",
        person_class_id: int | None = None,
    ):
        self.model = YOLO(model).to(device)
        if person_class_id is not None:
            self.person_class_id = person_class_id
        else:
            # Auto-detect class ID named 'person' (e.g. 11 for cityscapes semantic models, 0 for COCO)
            person_ids = [k for k, v in getattr(self.model, "names", {}).items() if str(v).lower() == "person"]
            self.person_class_id = person_ids[0] if person_ids else 11

    def predict(
        self,
        image: NumpyImage,
        imgsz: int = 1280,
        **kwargs,
    ) -> SegmentationResult:
        """
        Run semantic segmentation inference and return a standardized SegmentationResult.
        """

        # Run Ultralytics semantic inference
        results: list[Results] = self.model.predict(
            source=image,
            imgsz=imgsz,
            verbose=False,
            **kwargs,
        ) # type: ignore

        r = results[0]

        # Extract 2D class-ID mask
        if not hasattr(r, "semantic_mask") or r.semantic_mask is None:
            raise ValueError(
                "Model did not produce a semantic_mask. "
                "Ensure you pass a semantic segmentation model checkpoint (e.g., '*-sem.pt') "
                "rather than an instance segmentation ('*-seg.pt') or object detection model."
            )

        mask_data = r.semantic_mask.data
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()
        class_mask = mask_data.astype(np.int32)

        return SegmentationResult(
            mask=class_mask,
            class_names=self.model.names if hasattr(self.model, "names") else None,
        )

    def extract_roi(
        self,
        image: NumpyImage,
        fill_value: int = 0,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict semantic mask, extract ROI, and replace non-ROI pixels with fill_value.
        """
        # 1. Run prediction
        result = self.predict(image, **kwargs)

        # 2. Extract boolean mask for target classes
        roi_mask = result.get_binary_mask([self.person_class_id])

        # 3. Create masked image (default: black fill for non-ROI)
        masked_image = image.copy()
        masked_image[~roi_mask] = fill_value

        return roi_mask, masked_image
