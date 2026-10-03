from __future__ import annotations

from typing import Self

import cv2
import numpy as np


class NumpyImage(np.ndarray):
    """A validated np.ndarray subtype representing an image."""

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        if not data:
            raise ValueError("Input byte buffer is empty.")

        np_buf = np.frombuffer(data, dtype=np.uint8)
        bgr_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

        if bgr_img is None:
            raise ValueError("Invalid image file or unsupported format.")

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        return cls(rgb_img)

    def __new__(cls, input_array: np.ndarray) -> Self:
        arr = np.asanyarray(input_array)

        # 1. Parsing & invariant validation
        if arr.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {arr.ndim}D")

        if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Image expected 1, 3, or 4 channels, got {arr.shape[2]}"
            )

        valid_dtypes = (np.uint8, np.float32, np.float64)
        if arr.dtype not in valid_dtypes:
            raise ValueError(
                f"Image dtype must be one of {valid_dtypes}, got {arr.dtype}"
            )

        if np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any():
                raise ValueError("Image contains NaN values")
            if (arr < 0.0).any() or (arr > 1.0).any():
                raise ValueError("Float image values must be in [0.0, 1.0]")

        # 2. View casting into the subclass
        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj: object | None) -> None:
        """Called automatically when NumPy creates views or slices."""
        if obj is None:
            return

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def channels(self) -> int:
        return 1 if self.ndim == 2 else self.shape[2]
