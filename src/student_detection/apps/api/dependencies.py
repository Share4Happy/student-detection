from fastapi import Request

from .detector_pipeline import DetectorPipelineWorker

def get_detector_pipeline_worker(
    req: Request
) -> DetectorPipelineWorker:
    state = req.app.state

    if not hasattr(state, "detector_pipeline_worker"):
        raise RuntimeError("detector_pipeline_worker is not set in state")

    return state.detector_pipeline_worker
