from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """Represent an individual detected person item in the API response.

    Attributes:
        `box_id`: Identifier index of the detection box within the image.
        `bbox`: Bounding box pixel coordinates ordered as `[x1, y1, x2, y2]`.
        `confidence`: Detection confidence score in range `[0.0, 1.0]`.
        `depth`: Median relative depth value within person mask.
        `depth_std`: Standard deviation of relative depth values within person mask.
        `kept`: Flag indicating whether detection was retained after depth-NMS.
        `suppressed_by`: Identifier of suppressing box if suppressed, else `None`.
        `suppression_reason`: Detailed explanation if suppressed, else `None`.

    Methods:
        None.

    Examples:
        >>> item = DetectionItem(
        ...     box_id=0,
        ...     bbox=[10, 20, 100, 200],
        ...     confidence=0.95,
        ...     depth=0.5,
        ...     depth_std=0.02,
        ... )
        >>> item.kept
        True
    """

    box_id: int
    bbox: list[int] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float
    depth: float
    depth_std: float
    kept: bool = True
    suppressed_by: int | None = None
    suppression_reason: str | None = None


class PredictResponse(BaseModel):
    """Encapsulate the complete response payload returned by `/api/predict`.

    Attributes:
        `id`: Unique identifier tracking the prediction job.
        `people_count`: Final count of active, non-suppressed person detections.
        `detections`: Complete sequence of candidate and suppressed detections.
        `kept_count`: Total number of active person detections retained.
        `suppressed_count`: Total number of candidate boxes suppressed.

    Methods:
        None.

    Examples:
        >>> resp = PredictResponse(
        ...     id="req-1",
        ...     people_count=1,
        ...     detections=[],
        ...     kept_count=1,
        ...     suppressed_count=0,
        ... )
        >>> resp.people_count
        1
    """

    id: str
    people_count: int
    detections: list[DetectionItem]
    kept_count: int
    suppressed_count: int
