from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import PipelineDetection


class SingleDetectionPolicy(ABC):
    """Define the abstract base contract evaluated on a single person detection.

    Attributes:
        None.

    Methods:
        `__call__(person)`: Compute the threshold value for a detection.

    Examples:
        >>> class CustomPolicy(SingleDetectionPolicy):
        ...     def __call__(self, person: PipelineDetection) -> float:
        ...         return 0.5
        >>> policy = CustomPolicy()
        >>> isinstance(policy, SingleDetectionPolicy)
        True
    """

    @abstractmethod
    def __call__(self, person: PipelineDetection) -> float:
        """Compute the threshold value for a single person detection.

        Args:
            `person`: Person detection candidate containing spatial and depth metrics.

        Returns:
            Calculated threshold value as a `float`.

        Examples:
            >>> class DummyPolicy(SingleDetectionPolicy):
            ...     def __call__(self, person: PipelineDetection) -> float:
            ...         return 0.5
            >>> policy = DummyPolicy()
            >>> det = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
            >>> policy(det)
            0.5
        """


class PairwiseDetectionPolicy(ABC):
    """Define the abstract base contract evaluated between two person detections.

    Attributes:
        None.

    Methods:
        `__call__(person_a, person_b)`: Compute the threshold value for two detections.

    Examples:
        >>> class CustomPairwisePolicy(PairwiseDetectionPolicy):
        ...     def __call__(self, person_a: PipelineDetection, person_b: PipelineDetection) -> float:
        ...         return 0.2
        >>> policy = CustomPairwisePolicy()
        >>> isinstance(policy, PairwiseDetectionPolicy)
        True
    """

    @abstractmethod
    def __call__(
        self, person_a: PipelineDetection, person_b: PipelineDetection
    ) -> float:
        """Compute the threshold value between two candidate person detections.

        Args:
            `person_a`: First person detection candidate.
            `person_b`: Second person detection candidate.

        Returns:
            Calculated pairwise threshold value as a `float`.

        Examples:
            >>> class DummyPairPolicy(PairwiseDetectionPolicy):
            ...     def __call__(self, person_a: PipelineDetection, person_b: PipelineDetection) -> float:
            ...         return 0.2
            >>> policy = DummyPairPolicy()
            >>> det_a = PipelineDetection(0, [0, 0, 10, 10], 0.9, 0.5, 0.01)
            >>> det_b = PipelineDetection(1, [5, 5, 15, 15], 0.8, 0.6, 0.02)
            >>> policy(det_a, det_b)
            0.2
        """
