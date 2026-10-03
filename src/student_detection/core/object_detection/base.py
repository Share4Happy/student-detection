from abc import ABC, abstractmethod
from dataclasses import dataclass

from student_detection.common import NumpyImage


@dataclass
class BoundingBox:
    """Represents a single 2D bounding box detection."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str | None = None

    @property
    def xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

class ObjectDetectionModel(ABC):
    """Abstract interface for 2D object detection models."""

    @abstractmethod
    def predict(
        self,
        image: NumpyImage,
        conf_threshold: float = 0.25,
        **kwargs,
    ) -> list[BoundingBox]:
        """
        Run object detection inference.

        Args:
            image: Input image as a numpy array with shape (H, W, 3).
            conf_threshold: Minimum confidence score to retain a detection.
            target_classes: Optional list of class IDs to filter.
        Returns:
            List of detected BoundingBox objects.
        """
