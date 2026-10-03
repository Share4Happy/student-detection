from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from student_detection.common import NumpyImage


@dataclass
class SegmentationResult:
    """Represents the semantic segmentation output."""
    mask: np.ndarray  # 2D array (H, W) containing integer class IDs
    class_names: dict[int, str] | None = None

    def get_binary_mask(self, class_ids: list[int]) -> np.ndarray:
        """Returns a boolean mask where True indicates pixels matching target class IDs."""
        return np.isin(self.mask, class_ids)


class SemanticSegmentationModel(ABC):
    """Abstract interface for semantic segmentation models."""

    @abstractmethod
    def predict(
        self,
        image: NumpyImage,
        **kwargs
    ) -> SegmentationResult:
        """
        Run semantic segmentation inference.

        Args:
            image: Input image as a numpy array with shape (H, W, 3).
        Returns:
            SegmentationResult containing the 2D class-ID map.
        """

    @abstractmethod
    def extract_roi(
        self,
        image: NumpyImage,
        fill_value: int = 0,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract ROI mask and return masked image with background replaced.

        Args:
            image: Input image (H, W, 3).
            fill_value: Pixel value for non-ROI regions (default: 0 for black).
        Returns:
            Tuple of (boolean_mask [H, W], masked_image [H, W, 3]).
        """
