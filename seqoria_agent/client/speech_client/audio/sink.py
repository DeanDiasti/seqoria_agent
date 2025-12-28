"""Audio sink abstractions (Phase 2).

Sinks consume :class:`AudioChunk` buffers and render them to speakers,
transmit over WebRTC, or write to files.  The null sink is handy for tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from types import ModuleType
from contextlib import suppress

from seqoria_agent.client.speech_client.audio.source import _AsyncStreamContext  # type: ignore
from seqoria_agent.models.speech_models import AudioChunk


class BaseAudioSink(ABC):
    """Render or transmit assistant audio."""

    @abstractmethod
    async def consume(self, chunk: AudioChunk) -> None:  # noqa: D401
        """Process a single :class:`AudioChunk`.  Called for every TTS frame."""
        raise NotImplementedError


class NullSink(BaseAudioSink):
    """Drops all audio (for unit tests)."""

    async def consume(self, chunk: AudioChunk) -> None:  # pragma: no cover
        return 

# ---------------------------------------------------------------------------
# Concrete sink – speaker output
# ---------------------------------------------------------------------------

class SpeakerSink(BaseAudioSink):
    """Play audio chunks to the default speaker using *sounddevice*."""

    def __init__(self, *, sample_rate: int = 24_000, channels: int = 1) -> None:
        # Lazy import sounddevice
        try:
            import sounddevice as _sd  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover – tested separately
            raise ImportError("SpeakerSink requires the 'sounddevice' package - install via "
                              "'pip install \"seqoria-agent[speech]\"'") from exc

        self._sd: ModuleType = _sd  # stash reference
        self._sr = int(sample_rate)
        self._channels = int(channels)
        self._q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:  # noqa: D401
        loop = asyncio.get_running_loop()

        # Use RawOutputStream so we can write raw PCM bytes directly.
        stream = self._sd.RawOutputStream(channels=self._channels,
                                          samplerate=self._sr,
                                          dtype="int16")

        async with _AsyncStreamContext(stream, loop):
            while True:
                pcm = await self._q.get()
                # Convert to numpy array expected by sounddevice – but to avoid importing numpy
                # we rely on sounddevice accepting raw bytes when dtype is specified; if not, we
                # can convert in-place but keep simple here.
                try:
                    stream.write(pcm)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover – drop frames on error
                    pass

    async def consume(self, chunk: AudioChunk) -> None:  # pragma: no cover not used in tests
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        await self._q.put(chunk.pcm)

    # Provide graceful shutdown for orchestrator future phases
    async def close(self) -> None:  # noqa: D401
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None 
