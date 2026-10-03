from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import torch
import uvicorn
from fastapi import APIRouter, FastAPI

from student_detection.core import DetectorPipeline
from student_detection.core.depth_estimation import DepthAnythingV2
from student_detection.core.object_detection import YOLOPerson
from student_detection.core.policy import (
    FixedConfidencePolicy,
    FixedDepthThresholdPolicy,
    FixedIoAThresholdPolicy,
    FixedIoUThresholdPolicy,
)
from student_detection.core.semantic_segmentation import YOLOSemanticPerson

from .routes import predict
from .worker import DetectorPipelineWorker

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("StudentDetectionAPI")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def find_default_model(model_name: str) -> str:
    candidates = [
        PROJECT_ROOT / "data" / "models" / model_name,
        PROJECT_ROOT / "models" / model_name,
        Path("data/models") / model_name,
        Path("models") / model_name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(PROJECT_ROOT / "data" / "models" / model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sem_model_path = os.getenv("SEM_MODEL_PATH", find_default_model("yolo26x-sem.pt"))
    det_model_path = os.getenv("DET_MODEL_PATH", find_default_model("yolo26x.pt"))
    depth_checkpoint = os.getenv("DEPTH_CHECKPOINT", "depth-anything/Depth-Anything-V2-Small-hf")

    logger.info("Initializing models on device: %s", device)
    sem_model = YOLOSemanticPerson(model=sem_model_path, device=device)
    det_model = YOLOPerson(model=det_model_path, device=device)
    dep_model = DepthAnythingV2(model_path=depth_checkpoint, device=device)

    pipeline = DetectorPipeline(
        sem_model=sem_model,
        obj_model=det_model,
        dep_model=dep_model,
    )

    predict_fn = partial(
        pipeline.predict,
        conf_policy=FixedConfidencePolicy(0.25),
        iou_policy=FixedIoUThresholdPolicy(0.25),
        ioa_policy=FixedIoAThresholdPolicy(0.35),
        depth_policy=FixedDepthThresholdPolicy(0.010),
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

api_router = APIRouter(prefix="/api")
api_router.include_router(predict.api_router)

app.include_router(api_router)


def main():
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8080,
    )


if __name__ == "__main__":
    main()
