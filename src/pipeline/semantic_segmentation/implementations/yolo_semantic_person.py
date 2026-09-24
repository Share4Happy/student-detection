from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ....common import NumpyImage
from ..base import SegmentationResult, SemanticSegmentationModel


class YOLOSemanticPerson(SemanticSegmentationModel):
    """A YOLO semantic segmentation adapter that only segments person"""
    def __init__(
        self,
        model: str | Path = "yolo26n-sem.pt",
        device: Literal["cpu", "cuda"] = "cpu"
    ):
        self.model = YOLO(model).to(device)

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
        h, w = image.shape[:2]

        # Extract 2D class-ID mask
        if hasattr(r, "semantic_mask") and r.semantic_mask is not None:
            mask_data = r.semantic_mask.data
            if isinstance(mask_data, torch.Tensor):
                mask_data = mask_data.cpu().numpy()
            class_mask = mask_data.astype(np.int32)
        else:
            # Fallback if no mask is produced: treat entire image as background (255)
            class_mask = np.full((h, w), fill_value=255, dtype=np.int32)

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
        roi_mask = result.get_binary_mask([0]) # Id for person class is 11

        # 3. Create masked image (default: black fill for non-ROI)
        masked_image = image.copy()
        masked_image[~roi_mask] = fill_value

        return roi_mask, masked_image
