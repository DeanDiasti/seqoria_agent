"""Audio sink that writes AudioChunks to an aiortc MediaStreamTrack."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from seqoria_agent.models.speech_models import AudioChunk
from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink


class RTCSink(BaseAudioSink):
    """Consume chunks, encode as AudioFrame and write to track."""

    def __init__(self, track) -> None:  # type: ignore[no-any-unbound]
        try:
            from aiortc import MediaStreamTrack  # type: ignore
            from av import AudioFrame  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError("RTCSink requires 'aiortc' - install via 'pip install \"seqoria-agent[speech]\"'") from exc

        if not isinstance(track, MediaStreamTrack):
            raise TypeError("track must be aiortc.MediaStreamTrack")
        self._track = track
        self._AudioFrame = AudioFrame  # stash ref for fast access
        self._q: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=100)
        self._task: asyncio.Task | None = None

    async def _writer(self):  # noqa: D401
        while True:
            chunk = await self._q.get()
            # Construct a minimal AudioFrame and copy PCM data.
            frame = self._AudioFrame(samples=len(chunk.pcm) // 2,
                                     layout="mono",
                                     sample_rate=chunk.sample_rate)
            frame.planes[0].update(chunk.pcm)  # type: ignore[attr-defined]
            await self._track.write(frame)  # type: ignore[attr-defined]

    async def consume(self, chunk: AudioChunk) -> None:  # noqa: D401
        if self._task is None:
            self._task = asyncio.create_task(self._writer())
        await self._q.put(chunk)

    async def close(self) -> None:  # noqa: D401
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None 