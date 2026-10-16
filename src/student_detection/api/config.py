from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel


class ServerConfig(BaseModel):
    """Server network and hardware configuration."""

    HOST: ClassVar[str] = "0.0.0.0"
    PORT: ClassVar[int] = 8080
    DEVICE: ClassVar[Literal["auto", "cuda", "cpu"]] = "auto"


class ModelConfig(BaseModel):
    """Model checkpoints and weights."""

    DET_MODEL_PATH: ClassVar[str] = "data/models/yolo26m_finetuned_heads.pt"
    DEPTH_CHECKPOINT: ClassVar[str] = "depth-anything/Depth-Anything-V2-Small-hf"


class PipelineConfig(BaseModel):
    """Pipeline hyperparameters for filtering and Depth-NMS."""

    CONF_THRESHOLD: ClassVar[float] = 0.25
    IOU_THRESHOLD: ClassVar[float] = 0.25
    IOA_THRESHOLD: ClassVar[float] = 0.35
    DEPTH_THRESHOLD: ClassVar[float] = 0.010
    REFINE_BOXES: ClassVar[bool] = True
    IMGSZ: ClassVar[int] = 640
