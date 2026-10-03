from __future__ import annotations

from ...models import PipelineDetection
from ..base import PairwiseDetectionPolicy


class IoAThresholdPolicy(PairwiseDetectionPolicy):
    """Serve as the base class for IoA containment threshold policies.

    Attributes:
        None.

    Methods:
        `__call__(person_a, person_b)`: Compute the IoA threshold between two detections.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> class CustomIoA(IoAThresholdPolicy):
        ...     def __call__(self, person_a: PipelineDetection, person_b: PipelineDetection) -> float:
        ...         return 0.35
        >>> policy = CustomIoA()
        >>> isinstance(policy, IoAThresholdPolicy)
        True
    """


class FixedIoAThresholdPolicy(IoAThresholdPolicy):
    """Provide a constant IoA containment threshold between detection pairs.

    Attributes:
        `threshold`: Constant IoA containment threshold in range `[0.0, 1.0]`.

    Methods:
        `__call__(person_a, person_b)`: Return the constant IoA threshold.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> policy = FixedIoAThresholdPolicy(threshold=0.35)
        >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
        >>> det_b = PipelineDetection(1, [2, 2, 8, 8], 0.8, 0.5, 0.01)
        >>> policy(det_a, det_b)
        0.35
    """

    def __init__(self, threshold: float = 0.35) -> None:
        """Initialize the fixed IoA threshold policy.

        Args:
            `threshold`: Constant IoA containment cutoff. Defaults to `0.35`.

        Examples:
            >>> policy = FixedIoAThresholdPolicy(0.5)
            >>> policy.threshold
            0.5
        """
        self.threshold = threshold

    def __call__(
        self, person_a: PipelineDetection, person_b: PipelineDetection
    ) -> float:
        """Return the fixed IoA threshold.

        Args:
            `person_a`: First person detection candidate.
            `person_b`: Second person detection candidate.

        Returns:
            Configured constant IoA threshold as a `float`.

        Examples:
            >>> policy = FixedIoAThresholdPolicy(0.35)
            >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
            >>> det_b = PipelineDetection(1, [2, 2, 8, 8], 0.8, 0.5, 0.01)
            >>> policy(det_a, det_b)
            0.35
        """
        return self.threshold


__all__ = [
    "FixedIoAThresholdPolicy",
    "IoAThresholdPolicy",
]
