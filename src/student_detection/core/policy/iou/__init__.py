from __future__ import annotations

from ...models import PipelineDetection
from ..base import PairwiseDetectionPolicy


class IoUThresholdPolicy(PairwiseDetectionPolicy):
    """Serve as the base class for IoU spatial overlap threshold policies.

    Attributes:
        None.

    Methods:
        `__call__(person_a, person_b)`: Compute the IoU threshold between two detections.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> class CustomIoU(IoUThresholdPolicy):
        ...     def __call__(self, person_a: PipelineDetection, person_b: PipelineDetection) -> float:
        ...         return 0.3
        >>> policy = CustomIoU()
        >>> isinstance(policy, IoUThresholdPolicy)
        True
    """


class FixedIoUThresholdPolicy(IoUThresholdPolicy):
    """Provide a constant IoU spatial overlap threshold between detection pairs.

    Attributes:
        `threshold`: Constant IoU overlap threshold in range `[0.0, 1.0]`.

    Methods:
        `__call__(person_a, person_b)`: Return the constant IoU threshold.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> policy = FixedIoUThresholdPolicy(threshold=0.25)
        >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
        >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.5, 0.01)
        >>> policy(det_a, det_b)
        0.25
    """

    def __init__(self, threshold: float = 0.25) -> None:
        """Initialize the fixed IoU threshold policy.

        Args:
            `threshold`: Constant IoU overlap cutoff. Defaults to `0.25`.

        Examples:
            >>> policy = FixedIoUThresholdPolicy(0.4)
            >>> policy.threshold
            0.4
        """
        self.threshold = threshold

    def __call__(
        self, person_a: PipelineDetection, person_b: PipelineDetection
    ) -> float:
        """Return the fixed IoU threshold.

        Args:
            `person_a`: First person detection candidate.
            `person_b`: Second person detection candidate.

        Returns:
            Configured constant IoU threshold as a `float`.

        Examples:
            >>> policy = FixedIoUThresholdPolicy(0.25)
            >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
            >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.5, 0.01)
            >>> policy(det_a, det_b)
            0.25
        """
        return self.threshold


class DistanceScaledIoUPolicy(IoUThresholdPolicy):
    """Increase IoU threshold for distant groups to account for perspective overlap.

    Attributes:
        `base_threshold`: Baseline IoU threshold for close detections.
        `distance_boost`: Additional threshold factor scaled by average depth.

    Methods:
        `__call__(person_a, person_b)`: Return the distance-boosted IoU threshold.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> policy = DistanceScaledIoUPolicy(base_threshold=0.2, distance_boost=0.1)
        >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
        >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.5, 0.01)
        >>> policy(det_a, det_b)
        0.25
    """

    def __init__(
        self,
        base_threshold: float = 0.20,
        distance_boost: float = 0.15,
    ) -> None:
        """Initialize the distance-scaled IoU policy.

        Args:
            `base_threshold`: Minimum baseline IoU threshold. Defaults to `0.20`.
            `distance_boost`: Multiplier applied to average relative depth. Defaults to `0.15`.

        Examples:
            >>> policy = DistanceScaledIoUPolicy()
            >>> policy.base_threshold
            0.2
        """
        self.base_threshold = base_threshold
        self.distance_boost = distance_boost

    def __call__(
        self, person_a: PipelineDetection, person_b: PipelineDetection
    ) -> float:
        """Compute the distance-scaled IoU threshold between two detections.

        Args:
            `person_a`: First person detection candidate.
            `person_b`: Second person detection candidate.

        Returns:
            Calculated IoU threshold as `base_threshold + (distance_boost * avg_depth)`.

        Examples:
            >>> policy = DistanceScaledIoUPolicy(base_threshold=0.2, distance_boost=0.2)
            >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.4, 0.01)
            >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.6, 0.01)
            >>> policy(det_a, det_b)
            0.3
        """
        avg_depth = (person_a.depth + person_b.depth) / 2.0
        return self.base_threshold + (self.distance_boost * avg_depth)


__all__ = [
    "DistanceScaledIoUPolicy",
    "FixedIoUThresholdPolicy",
    "IoUThresholdPolicy",
]
