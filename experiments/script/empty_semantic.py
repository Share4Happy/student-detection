from __future__ import annotations

import numpy as np
from student_detection.common import NumpyImage
from student_detection.core.semantic_segmentation.base import (
    SegmentationResult,
    SemanticSegmentationModel,
)


class EmptyYOLOSemanticPerson(SemanticSegmentationModel):
    """Pass-through semantic segmentation adapter for DetectorPipeline.

    Bypasses semantic background suppression while satisfying the SemanticSegmentationModel
    interface. Treats the entire input frame as foreground person ROI, allowing
    DetectorPipeline to run with object detection and depth estimation alone.
    """

    def __init__(self, person_class_id: int = 1) -> None:
        self.person_class_id = person_class_id

    def predict(self, image: NumpyImage, **_kwargs) -> SegmentationResult:
        h, w = image.shape[:2]
        mask = np.full((h, w), self.person_class_id, dtype=np.int32)
        return SegmentationResult(mask=mask, class_names={self.person_class_id: "person"})

    def extract_roi(
        self,
        image: NumpyImage,
        fill_value: int = 0,
        **_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        roi_mask = np.ones((h, w), dtype=bool)
        # Returns unmasked image copy and full-frame boolean mask
        return roi_mask, image.copy()
