from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PipelineDetection:
    """Represent an individual person detection candidate in the pipeline.

    Attributes:
        `box_id`: Unique integer identifier for this detection within a frame.
        `bbox`: Bounding box pixel coordinates ordered as `[x1, y1, x2, y2]`.
        `confidence`: Detection confidence score in range `[0.0, 1.0]`.
        `depth`: Median relative depth value within the detection mask.
        `depth_std`: Standard deviation of relative depth values within the mask.
        `kept`: Boolean flag indicating whether this detection survived suppression.
        `suppressed_by`: Identifier of the candidate box that suppressed this box.
        `suppression_reason`: Descriptive explanation for why this box was suppressed.

    Methods:
        None.

    Examples:
        >>> detection = PipelineDetection(
        ...     box_id=0,
        ...     bbox=[100, 150, 200, 350],
        ...     confidence=0.88,
        ...     depth=0.45,
        ...     depth_std=0.03,
        ... )
        >>> detection.kept
        True
    """

    box_id: int
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    depth: float
    depth_std: float
    kept: bool = True
    suppressed_by: int | None = None
    suppression_reason: str | None = None


@dataclass
class PipelineResult:
    """Store the complete output of person detection and depth-aware counting.

    Attributes:
        `detections`: Complete sequence of candidate and suppressed detections.
        `people_count`: Final count of active, non-suppressed person detections.
        `sem_mask`: Binary semantic segmentation mask of shape `(H, W)`.
        `depth_map`: Normalized relative depth array of shape `(H, W)`.

    Methods:
        None.

    Examples:
        >>> result = PipelineResult(
        ...     detections=[],
        ...     people_count=0,
        ...     sem_mask=np.zeros((10, 10), dtype=np.uint8),
        ...     depth_map=np.zeros((10, 10), dtype=np.float32),
        ... )
        >>> result.people_count
        0
    """

    detections: list[PipelineDetection]
    people_count: int
    sem_mask: np.ndarray
    depth_map: np.ndarray
