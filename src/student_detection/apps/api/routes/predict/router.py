from __future__ import annotations

import uuid
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from student_detection.common import NumpyImage
from ...dependencies import get_detector_pipeline_worker
from ...detector_pipeline import DetectorPipelineWorker, Item
from .schemas import DetectionItem, PredictResponse

router = APIRouter(prefix="/predict")


@router.post(path="", response_model=PredictResponse)
async def predict(
    image: UploadFile,
    worker: DetectorPipelineWorker = Depends(get_detector_pipeline_worker),
) -> PredictResponse:
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG, PNG, and WebP images are allowed.",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    nparr = np.frombuffer(contents, np.uint8)
    bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr_img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    numpy_image = NumpyImage(bgr_img)
    item_id = str(uuid.uuid4())
    item = Item(id=item_id, image=numpy_image)

    res = await worker.predict_async(item)
    if not res.success or res.result is None:
        raise HTTPException(status_code=500, detail=f"Inference failed: {res.error}")

    detections = [
        DetectionItem(
            box_id=d.box_id,
            bbox=d.bbox,
            confidence=d.confidence,
            depth=d.depth,
            depth_std=d.depth_std,
            kept=d.kept,
            suppressed_by=d.suppressed_by,
            suppression_reason=d.suppression_reason,
        )
        for d in res.result.detections
    ]

    return PredictResponse(
        id=res.id,
        people_count=res.result.people_count,
        detections=detections,
        kept_count=res.result.people_count,
        suppressed_count=len(detections) - res.result.people_count,
    )
