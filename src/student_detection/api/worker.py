from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

from student_detection.common import NumpyImage
from student_detection.core import PipelineResult


@dataclass
class Item:
    """Encapsulate an inference request item for the pipeline worker.

    Attributes:
        `id`: Unique identifier tracking the prediction request.
        `image`: Input image formatted as a `NumpyImage`.

    Methods:
        None.

    Examples:
        >>> from student_detection.common import NumpyImage
        >>> import numpy as np
        >>> item = Item(id="req-1", image=NumpyImage(np.zeros((10, 10, 3), dtype=np.uint8)))
        >>> item.id
        'req-1'
    """

    id: str
    image: NumpyImage


@dataclass
class Result:
    """Encapsulate the completion result or failure of an inference item.

    Attributes:
        `id`: Identifier corresponding to the processed `Item`.
        `result`: Computed `PipelineResult` when inference succeeds, else `None`.
        `error`: Error message describing failure when unsuccessful, else `None`.
        `success`: Boolean indicating whether inference completed successfully.

    Methods:
        None.

    Examples:
        >>> res = Result(id="req-1", success=True)
        >>> res.success
        True
    """

    id: str
    result: PipelineResult | None = None
    error: str | None = None
    success: bool = True


@dataclass
class _WorkTask:
    """Bind an input item with its calling event loop and delivery future.

    Attributes:
        `item`: Input image item to process.
        `future`: Asyncio future resolved with the final `Result`.
        `loop`: Event loop instance originating the request.
    """

    item: Item
    future: asyncio.Future[Result]
    loop: asyncio.AbstractEventLoop


class DetectorPipelineWorker:
    """Coordinate asynchronous inference dispatching across dedicated worker threads.

    Attributes:
        None.

    Methods:
        `start()`: Start background thread and begin processing items.
        `stop(timeout=10.0)`: Gracefully terminate background worker thread.
        `predict_async(item)`: Submit an `Item` asynchronously and await the `Result`.

    Examples:
        >>> def fake_predict(img):
        ...     from student_detection.core import PipelineResult
        ...     import numpy as np
        ...     return PipelineResult([], 0, np.zeros((1, 1)), np.zeros((1, 1)))
        >>> worker = DetectorPipelineWorker(predict_fn=fake_predict)
        >>> worker.start()
        >>> worker.stop()
    """

    def __init__(
        self,
        predict_fn: Callable[[NumpyImage], PipelineResult],
        max_queue_size: int = 1000,
    ) -> None:
        """Initialize the pipeline worker instance.

        Args:
            `predict_fn`: Synchronous callable executing pipeline prediction.
            `max_queue_size`: Maximum allowed depth of pending requests. Defaults to `1000`.

        Examples:
            >>> worker = DetectorPipelineWorker(predict_fn=lambda img: None)
            >>> worker is not None
            True
        """
        self._predict_fn = predict_fn
        self._process_queue: queue.Queue[_WorkTask | None] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._bg_thread: threading.Thread | None = None

    def _processing(self) -> None:
        """Process incoming queue items sequentially in the background thread."""
        while self._running:
            try:
                task = self._process_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Shutdown sentinel received
            if task is None:
                self._process_queue.task_done()
                break

            item = task.item
            try:
                pred = self._predict_fn(item.image)
                output = Result(
                    id=item.id,
                    result=pred,
                    success=True,
                )
            except Exception as exc:
                output = Result(
                    id=item.id,
                    result=None,
                    error=str(exc),
                    success=False,
                )
            finally:
                self._process_queue.task_done()

            # Deliver Result back to caller's async event loop
            if not task.future.cancelled():
                task.loop.call_soon_threadsafe(task.future.set_result, output)

    def start(self) -> None:
        """Start the background worker thread.

        Examples:
            >>> worker = DetectorPipelineWorker(predict_fn=lambda img: None)
            >>> worker.start()
            >>> worker.stop()
        """
        if self._running:
            return
        self._running = True
        self._bg_thread = threading.Thread(
            target=self._processing,
            daemon=True,
            name="DetectorPipelineWorkerThread",
        )
        self._bg_thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the background worker thread cleanly.

        Args:
            `timeout`: Maximum duration in seconds to wait for thread join. Defaults to `10.0`.

        Examples:
            >>> worker = DetectorPipelineWorker(predict_fn=lambda img: None)
            >>> worker.start()
            >>> worker.stop(timeout=1.0)
        """
        if not self._running:
            return
        self._running = False
        self._process_queue.put(None)  # Wake up thread to exit
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=timeout)
        self._bg_thread = None

    async def predict_async(self, item: Item) -> Result:
        """Submit an item to the background worker and await its result.

        Args:
            `item`: Input data item wrapping identifier and image.

        Returns:
            Resolved `Result` containing prediction output or error string.

        Raises:
            `RuntimeError`: If worker is not currently active and running.

        Examples:
            >>> # result = await worker.predict_async(item)
            ... pass
        """
        if not self._running:
            raise RuntimeError("DetectorPipelineWorker is not running.")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Result] = loop.create_future()
        task = _WorkTask(item=item, future=future, loop=loop)

        self._process_queue.put_nowait(task)
        return await future

    def __enter__(self) -> Self:
        """Enter runtime context and start worker thread.

        Returns:
            Active `DetectorPipelineWorker` instance.

        Examples:
            >>> with DetectorPipelineWorker(predict_fn=lambda img: None) as w:
            ...     w._running
            True
        """
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit runtime context and stop worker thread cleanly.

        Args:
            `*_args`: Forwarded exception details from runtime context.

        Examples:
            >>> with DetectorPipelineWorker(predict_fn=lambda img: None) as w:
            ...     pass
        """
        self.stop()
