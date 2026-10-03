from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from student_detection.common import NumpyImage


@dataclass
class DepthResult:
    """Represents the estimated depth map output."""
    depth_map: np.ndarray  # 2D float array (H, W) with relative or metric depth values
    is_metric: bool = False

    def get_normalized(self) -> np.ndarray:
        """Returns a normalized depth map scaled to range [0.0, 1.0]."""
        d_min = float(self.depth_map.min())
        d_max = float(self.depth_map.max())
        if d_max > d_min:
            return (self.depth_map - d_min) / (d_max - d_min)
        return np.zeros_like(self.depth_map, dtype=np.float32)

class DepthEstimationModel(ABC):
    """Abstract interface for monocular depth estimation models."""

    @abstractmethod
    def predict(self, image: NumpyImage, **kwargs) -> DepthResult:
        """
        Estimate dense depth map from an input image.

        Args:
            image: Input image as a numpy array with shape (H, W, 3).
        Returns:
            DepthResult containing the estimated depth map.
        """
