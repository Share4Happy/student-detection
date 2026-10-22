from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import torch
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from student_detection.core import DetectorPipeline
from student_detection.core.depth_estimation import DepthAnythingV2
from student_detection.core.object_detection import YOLOFineTunedSingleClass
from student_detection.core.policy import (
    FixedConfidencePolicy,
    FixedDepthThresholdPolicy,
    FixedIoAThresholdPolicy,
    FixedIoUThresholdPolicy,
)
from student_detection.core.semantic_segmentation import EmptyYOLOSemanticPerson

from .config import ModelConfig, PipelineConfig, ServerConfig
from .routes import predict
from .worker import DetectorPipelineWorker

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("StudentDetectionAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = ServerConfig.DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    det_model_path = os.getenv("DET_MODEL_PATH", ModelConfig.DET_MODEL_PATH)
    depth_checkpoint = os.getenv("DEPTH_CHECKPOINT", ModelConfig.DEPTH_CHECKPOINT)

    logger.info("Initializing models on device: %s", device)
    sem_model = EmptyYOLOSemanticPerson()
    det_model = YOLOFineTunedSingleClass(model=det_model_path, device=device)
    dep_model = DepthAnythingV2(model_path=depth_checkpoint, device=device)

    pipeline = DetectorPipeline(
        sem_model=sem_model,
        obj_model=det_model,
        dep_model=dep_model,
    )

    predict_fn = partial(
        pipeline.predict,
        conf_policy=FixedConfidencePolicy(PipelineConfig.CONF_THRESHOLD),
        iou_policy=FixedIoUThresholdPolicy(PipelineConfig.IOU_THRESHOLD),
        ioa_policy=FixedIoAThresholdPolicy(PipelineConfig.IOA_THRESHOLD),
        depth_policy=FixedDepthThresholdPolicy(PipelineConfig.DEPTH_THRESHOLD),
        refine_boxes=PipelineConfig.REFINE_BOXES,
        imgsz=PipelineConfig.IMGSZ,
    )
    worker = DetectorPipelineWorker(predict_fn=predict_fn)
    worker.start()
    app.state.detector_pipeline_worker = worker
    logger.info("DetectorPipelineWorker started and ready.")

    try:
        yield
    finally:
        logger.info("Stopping DetectorPipelineWorker...")
        worker.stop()
        logger.info("DetectorPipelineWorker stopped.")


app = FastAPI(
    title="Student Detection API",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "data" / "samples" / "classroom.png"

api_router = APIRouter(prefix="/api")
api_router.include_router(predict.api_router)


@api_router.get("/info")
async def get_server_info():
    """Return active server, model, and pipeline configurations."""
    return {
        "server": {
            "host": ServerConfig.HOST,
            "port": ServerConfig.PORT,
            "device": ServerConfig.DEVICE,
        },
        "model": {
            "det_model_path": ModelConfig.DET_MODEL_PATH,
            "depth_checkpoint": ModelConfig.DEPTH_CHECKPOINT,
        },
        "pipeline": {
            "conf_threshold": PipelineConfig.CONF_THRESHOLD,
            "iou_threshold": PipelineConfig.IOU_THRESHOLD,
            "ioa_threshold": PipelineConfig.IOA_THRESHOLD,
            "depth_threshold": PipelineConfig.DEPTH_THRESHOLD,
            "refine_boxes": PipelineConfig.REFINE_BOXES,
            "imgsz": PipelineConfig.IMGSZ,
        },
    }


@api_router.get("/sample-image")
async def get_sample_image():
    """Return bundled sample image for quick UI testing."""
    if not SAMPLE_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample image not found on server.")
    return FileResponse(SAMPLE_IMAGE_PATH, media_type="image/png")


app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """Serve the frontend test inspection lab UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend UI template not found.")
    return FileResponse(index_file, media_type="text/html")


def main():
    uvicorn.run(
        app=app,
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
    )


if __name__ == "__main__":
    main()
