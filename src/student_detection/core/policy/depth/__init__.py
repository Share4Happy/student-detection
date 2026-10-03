from __future__ import annotations

from ...models import PipelineDetection
from ..base import PairwiseDetectionPolicy


class DepthThresholdPolicy(PairwiseDetectionPolicy):
    """Serve as the base class for depth difference threshold policies.

    Attributes:
        None.

    Methods:
        `__call__(person_a, person_b)`: Compute the depth threshold between two detections.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> class CustomDepth(DepthThresholdPolicy):
        ...     def __call__(self, person_a: PipelineDetection, person_b: PipelineDetection) -> float:
        ...         return 0.01
        >>> policy = CustomDepth()
        >>> isinstance(policy, DepthThresholdPolicy)
        True
    """


class FixedDepthThresholdPolicy(DepthThresholdPolicy):
    """Provide a constant depth difference threshold between detection pairs.

    Attributes:
        `threshold`: Constant depth difference threshold.

    Methods:
        `__call__(person_a, person_b)`: Return the constant depth difference threshold.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> policy = FixedDepthThresholdPolicy(threshold=0.01)
        >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
        >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.52, 0.01)
        >>> policy(det_a, det_b)
        0.01
    """

    def __init__(self, threshold: float = 0.010) -> None:
        """Initialize the fixed depth threshold policy.

        Args:
            `threshold`: Constant relative depth difference cutoff. Defaults to `0.010`.

        Examples:
            >>> policy = FixedDepthThresholdPolicy(0.02)
            >>> policy.threshold
            0.02
        """
        self.threshold = threshold

    def __call__(
        self, person_a: PipelineDetection, person_b: PipelineDetection
    ) -> float:
        """Return the fixed depth difference threshold.

        Args:
            `person_a`: First person detection candidate.
            `person_b`: Second person detection candidate.

        Returns:
            Configured constant depth threshold as a `float`.

        Examples:
            >>> policy = FixedDepthThresholdPolicy(0.01)
            >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
            >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.52, 0.01)
            >>> policy(det_a, det_b)
            0.01
        """
        return self.threshold


class DistanceDecayDepthPolicy(DepthThresholdPolicy):
    """Dynamically scale depth threshold down as distance from camera increases.

    Attributes:
        `base_threshold`: Baseline depth threshold for close detections.
        `min_threshold`: Minimum allowable threshold floor for distant detections.
        `decay_factor`: Rate of linear threshold decay relative to average depth.

    Methods:
        `__call__(person_a, person_b)`: Return the distance-decayed depth threshold.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> policy = DistanceDecayDepthPolicy(base_threshold=0.02, min_threshold=0.005, decay_factor=0.5)
        >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.6, 0.01)
        >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.6, 0.01)
        >>> policy(det_a, det_b)
        0.014
    """

    def __init__(
        self,
        base_threshold: float = 0.015,
        min_threshold: float = 0.004,
        decay_factor: float = 0.70,
    ) -> None:
        """Initialize the distance decay depth policy.

        Args:
            `base_threshold`: Baseline depth difference cutoff. Defaults to `0.015`.
            `min_threshold`: Minimum allowable threshold floor. Defaults to `0.004`.
            `decay_factor`: Linear decay rate multiplier. Defaults to `0.70`.

        Examples:
            >>> policy = DistanceDecayDepthPolicy()
            >>> policy.base_threshold
            0.015
        """
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.decay_factor = decay_factor

    def __call__(
        self, person_a: PipelineDetection, person_b: PipelineDetection
    ) -> float:
        """Compute the distance-decayed depth threshold between two detections.

        Args:
            `person_a`: First person detection candidate.
            `person_b`: Second person detection candidate.

        Returns:
            Calculated depth threshold as a `float`, clamped above `min_threshold`.

        Examples:
            >>> policy = DistanceDecayDepthPolicy(base_threshold=0.01, min_threshold=0.002, decay_factor=0.5)
            >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.4, 0.01)
            >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.4, 0.01)
            >>> policy(det_a, det_b)
            0.008
        """
        avg_depth = (person_a.depth + person_b.depth) / 2.0
        scaled = self.base_threshold * (1.0 - self.decay_factor * avg_depth)
        return max(self.min_threshold, scaled)


__all__ = [
    "DepthThresholdPolicy",
    "DistanceDecayDepthPolicy",
    "FixedDepthThresholdPolicy",
]
