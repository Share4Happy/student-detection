from pathlib import Path
from typing import Literal

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from ....common import NumpyImage
from ..base import DepthEstimationModel, DepthResult


class DepthAnythingV2(DepthEstimationModel):
    def __init__(
        self,
        model_path: str | Path = "depth-anything/Depth-Anything-V2-Small-hf",
        device: Literal["cuda", "cpu"] = "cpu"
    ):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path).to(device)
        self.model.eval()

    def predict(
        self,
        image: NumpyImage,
        **_kwargs
    ) -> DepthResult:
        """
        Estimate dense relative depth map and resize it to original image dimensions.

        Args:
            image: BGR or RGB image as numpy array of shape (H, W, 3).
        Returns:
            DepthResult containing the 2D depth map.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        h, w = image.shape[:2]

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Preprocess input
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate depth back to original image resolution
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        depth_map = prediction.squeeze().cpu().numpy()

        return DepthResult(
            depth_map=depth_map,
            is_metric=False,  # DA2 outputs affine-invariant relative depth
        )
