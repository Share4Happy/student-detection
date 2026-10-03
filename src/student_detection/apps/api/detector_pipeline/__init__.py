from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

from student_detection.common import NumpyImage
from student_detection.pipeline import PipelineResult


@dataclass
class Item:
    id: str
    image: NumpyImage


@dataclass
class Result:
    id: str
    result: PipelineResult | None = None
    error: str | None = None
    success: bool = True


@dataclass
class _WorkTask:
    item: Item
    future: asyncio.Future[Result]
    loop: asyncio.AbstractEventLoop


class DetectorPipelineWorker:
    def __init__(
        self,
        predict_fn: Callable[[NumpyImage], PipelineResult],
        max_queue_size: int = 1000,
    ):
        self._predict_fn = predict_fn
        self._process_queue: queue.Queue[_WorkTask | None] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._bg_thread: threading.Thread | None = None

    def _processing(self) -> None:
        """Worker thread processing items sequentially."""
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
        """Start the background worker thread."""
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
        """Stop the background worker thread cleanly."""
        if not self._running:
            return
        self._running = False
        self._process_queue.put(None)  # Wake up thread to exit
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=timeout)
        self._bg_thread = None

    async def predict_async(self, item: Item) -> Result:
        """Async entrypoint for FastAPI endpoints."""
        if not self._running:
            raise RuntimeError("DetectorPipelineWorker is not running.")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Result] = loop.create_future()
        task = _WorkTask(item=item, future=future, loop=loop)

        self._process_queue.put_nowait(task)
        return await future

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
