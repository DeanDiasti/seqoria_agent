from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any


BackgroundTask = Callable[[], Awaitable[Any]]


class BackgroundWorker:
    """Small async queue runner for opportunistic memory processing."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._in_flight = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    @property
    def is_processing(self) -> bool:
        return self._in_flight > 0 or not self._queue.empty()

    @property
    def is_running(self) -> bool:
        return self._running

    def submit(self, task_factory: BackgroundTask) -> None:
        self._queue.put_nowait(task_factory)

    async def flush(self) -> None:
        await self._queue.join()

    async def _run_loop(self) -> None:
        while self._running:
            task_factory = await self._queue.get()
            self._in_flight += 1
            try:
                await task_factory()
            except Exception:
                # Memory processing failures should not break user-facing flow.
                pass
            finally:
                self._in_flight = max(0, self._in_flight - 1)
                self._queue.task_done()

