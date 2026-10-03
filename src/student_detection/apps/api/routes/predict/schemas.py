from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    box_id: int
    bbox: list[int] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float
    depth: float
    depth_std: float
    kept: bool = True
    suppressed_by: int | None = None
    suppression_reason: str | None = None


class PredictResponse(BaseModel):
    id: str
    people_count: int
    detections: list[DetectionItem]
    kept_count: int
    suppressed_count: int
