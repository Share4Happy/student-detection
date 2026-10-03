from fastapi import Request

from .detector_pipeline import DetectorPipelineWorker


def get_detector_pipeline_worker(
    req: Request,
) -> DetectorPipelineWorker:
    """Retrieve the pipeline worker instance from FastAPI application state.

    Args:
        `req`: Incoming FastAPI `Request` object carrying application state.

    Returns:
        Configured `DetectorPipelineWorker` singleton instance.

    Raises:
        `RuntimeError`: If `detector_pipeline_worker` is not configured on `req.app.state`.

    Examples:
        >>> # worker = get_detector_pipeline_worker(request)
        ... pass
    """
    state = req.app.state

    if not hasattr(state, "detector_pipeline_worker"):
        raise RuntimeError("detector_pipeline_worker is not set in state")

    return state.detector_pipeline_worker
