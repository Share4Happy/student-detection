from __future__ import annotations

from ...models import PipelineDetection
from ..base import SingleDetectionPolicy


class ConfidencePolicy(SingleDetectionPolicy):
    """Serve as the base class for person confidence threshold policies.

    Attributes:
        None.

    Methods:
        `__call__(person)`: Compute the required confidence cutoff for a detection.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> class CustomConf(ConfidencePolicy):
        ...     def __call__(self, person: PipelineDetection) -> float:
        ...         return 0.4
        >>> policy = CustomConf()
        >>> isinstance(policy, ConfidencePolicy)
        True
    """


class FixedConfidencePolicy(ConfidencePolicy):
    """Provide a constant confidence threshold for all candidate detections.

    Attributes:
        `threshold`: Constant minimum confidence score in range `[0.0, 1.0]`.

    Methods:
        `__call__(person)`: Return the constant confidence threshold.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> policy = FixedConfidencePolicy(threshold=0.3)
        >>> det = PipelineDetection(0, [0, 0, 10, 10], 0.8, 0.5, 0.01)
        >>> policy(det)
        0.3
    """

    def __init__(self, threshold: float = 0.25) -> None:
        """Initialize the fixed confidence policy.

        Args:
            `threshold`: Constant minimum confidence score. Defaults to `0.25`.

        Examples:
            >>> policy = FixedConfidencePolicy(0.5)
            >>> policy.threshold
            0.5
        """
        self.threshold = threshold

    def __call__(self, person: PipelineDetection) -> float:
        """Return the fixed confidence threshold for the candidate detection.

        Args:
            `person`: Person detection candidate to evaluate.

        Returns:
            Configured constant confidence threshold as a `float`.

        Examples:
            >>> policy = FixedConfidencePolicy(0.3)
            >>> det = PipelineDetection(0, [0, 0, 10, 10], 0.8, 0.5, 0.01)
            >>> policy(det)
            0.3
        """
        return self.threshold


class DistanceScaledConfidencePolicy(ConfidencePolicy):
    """Scale required confidence based on person distance from the camera.

    Attributes:
        `near_threshold`: Required confidence score for close objects.
        `far_threshold`: Relaxed confidence score for distant objects.
        `depth_cutoff`: Relative depth boundary separating near and far objects.

    Methods:
        `__call__(person)`: Return the distance-adapted confidence threshold.

    Examples:
        >>> from student_detection.core.models import PipelineDetection
        >>> policy = DistanceScaledConfidencePolicy(near_threshold=0.3, far_threshold=0.15, depth_cutoff=0.5)
        >>> near_det = PipelineDetection(0, [0, 0, 10, 10], 0.8, 0.2, 0.01)
        >>> far_det = PipelineDetection(1, [0, 0, 10, 10], 0.8, 0.7, 0.01)
        >>> policy(near_det)
        0.3
        >>> policy(far_det)
        0.15
    """

    def __init__(
        self,
        near_threshold: float = 0.30,
        far_threshold: float = 0.15,
        depth_cutoff: float = 0.60,
    ) -> None:
        """Initialize the distance-scaled confidence policy.

        Args:
            `near_threshold`: Minimum confidence required for near objects. Defaults to `0.30`.
            `far_threshold`: Minimum confidence required for far objects. Defaults to `0.15`.
            `depth_cutoff`: Relative depth boundary value. Defaults to `0.60`.

        Examples:
            >>> policy = DistanceScaledConfidencePolicy()
            >>> policy.near_threshold
            0.3
        """
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.depth_cutoff = depth_cutoff

    def __call__(self, person: PipelineDetection) -> float:
        """Calculate the confidence threshold based on detection depth.

        Args:
            `person`: Person detection candidate to evaluate.

        Returns:
            `far_threshold` if `person.depth >= depth_cutoff`, else `near_threshold`.

        Examples:
            >>> policy = DistanceScaledConfidencePolicy(depth_cutoff=0.5)
            >>> det = PipelineDetection(0, [0, 0, 10, 10], 0.8, 0.8, 0.01)
            >>> policy(det)
            0.15
        """
        return self.far_threshold if person.depth >= self.depth_cutoff else self.near_threshold


__all__ = [
    "ConfidencePolicy",
    "DistanceScaledConfidencePolicy",
    "FixedConfidencePolicy",
]
