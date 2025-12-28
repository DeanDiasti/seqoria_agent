"""Audio source abstractions (Phase 2).

Sources yield raw :class:`AudioChunk` objects from microphones, files or
network streams.  This base interface allows the rest of the speech pipeline
(STT, AI, TTS) to remain agnostic to the actual audio input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Self
import asyncio
import wave
from pathlib import Path
from types import ModuleType

from seqoria_agent.models.speech_models import AudioChunk
from seqoria_agent.client.speech_client.utils import SilenceGate


class BaseAudioSource(ABC):
    """Yield raw audio frames from an input device or file."""

    @abstractmethod
    async def frames(self) -> AsyncIterator[AudioChunk]:  # noqa: D401
        """Async generator producing :class:`AudioChunk` buffers."""
        raise NotImplementedError


class NullSource(BaseAudioSource):
    """Test-double that produces **no** frames (useful for unit tests)."""

    async def frames(self):  # pragma: no cover
        if False:  # never executes but satisfies generator typing
            yield  # type: ignore[misc] 


# Concrete sources – file & microphone
class FileSource(BaseAudioSource):
    """Yield audio frames from a WAV file (blocking I/O moved off-thread)."""

    def __init__(self, wav_path: str | Path, *, chunk_ms: int = 20) -> None:
        self._path = Path(wav_path)
        if not self._path.is_file():
            raise FileNotFoundError(self._path)
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        self._chunk_ms = int(chunk_ms)

    async def frames(self) -> AsyncIterator[AudioChunk]:  # noqa: D401
        loop = asyncio.get_running_loop()
        # Open the wav file in a thread to avoid blocking
        wf = await loop.run_in_executor(None, wave.open, str(object=self._path), "rb")

        try:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            bytes_per_frame = sampwidth * channels
            frames_per_chunk = int(sample_rate * self._chunk_ms / 1000)

            async def _read_chunk():  # helper to move blocking I/O off-thread
                return await loop.run_in_executor(None, wf.readframes, frames_per_chunk)

            while True:
                data: bytes = await _read_chunk()
                if not data:
                    break
                yield AudioChunk(pcm=data, sample_rate=sample_rate, channels=channels)
        finally:
            await loop.run_in_executor(None, wf.close)


class MicSource(BaseAudioSource):
    """Capture audio from the default system microphone (async-friendly)."""

    def __init__(self, *, sample_rate: int = 16_000, chunk_ms: int = 20) -> None:
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        self._sr = int(sample_rate)
        self._chunk_ms = int(chunk_ms)

        # Lazy import – raise helpful error when sounddevice missing
        try:
            import sounddevice as _sd  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover – tested separately
            raise ImportError("MicSource requires the 'sounddevice' package - install via "
                              "'pip install \"seqoria-agent[speech]\"'") from exc

        self._sd: ModuleType = _sd  # store reference for later use

    async def frames(self) -> AsyncIterator[AudioChunk]:  # noqa: D401
        bytes_per_sample = 2  # int16
        frames_per_chunk = int(self._sr * self._chunk_ms / 1000)
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        def _callback(indata, _frames, _time, _status):  # noqa: D401, ANN001
            """sounddevice callback - *indata* may be NumPy array **or** bytes.

            The real sounddevice stream passes a ``numpy.ndarray`` - however
            the unit tests inject raw *bytes* to avoid the NumPy dependency.
            We therefore accept either type and normalise to ``bytes`` before
            enqueuing.
            """

            # Normalise to raw PCM bytes
            if hasattr(indata, "tobytes"):
                # NumPy path – fast conversion without copy
                pcm_bytes = indata.tobytes()
            elif isinstance(indata, (bytes, bytearray)):
                pcm_bytes = bytes(indata)
            else:  # pragma: no cover – unexpected input type
                return

            # Enqueue in **event-loop thread** to avoid cross-thread access
            def _enqueue(data: bytes) -> None:  # noqa: D401
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass  # drop if queue saturated

            try:
                loop.call_soon_threadsafe(_enqueue, pcm_bytes)
            except RuntimeError:  # pragma: no cover – event loop closed
                pass
            
        stream = self._sd.InputStream(channels=1,
                                      samplerate=self._sr,
                                      blocksize=frames_per_chunk,
                                      dtype="int16",
                                      callback=_callback)

        loop = asyncio.get_running_loop()

        # Run the blocking stream in default executor so context manager doesn't block event loop
        async with _AsyncStreamContext(stream_ctx=stream, loop=loop):
            while True:
                pcm = await q.get()
                yield AudioChunk(pcm=pcm, sample_rate=self._sr, channels=1)

    async def clip_frames(
        self,
        clip_secs: float = 5.0,
        *,
        vad_threshold_ms: int = 600,
        energy_gate: float = 30.0,
    ) -> AsyncIterator[AudioChunk]:
        """Yield multi-second audio clips captured from the microphone.

        This helper aggregates the short ``chunk_ms``-sized frames produced by
        :pymeth:`frames` until *clip_secs* seconds worth of PCM has been
        collected and then yields a single :class:`~seqoria_agent.models.speech_models.AudioChunk`.
        Any excess audio beyond the requested duration is carried over to the
        next clip to avoid data loss.

        Parameters
        ----------
        clip_secs:
            Desired length of each emitted clip in **seconds**. Must be
            positive. Defaults to 5 seconds.
        """
        if clip_secs <= 0:
            raise ValueError("clip_secs must be positive")

        bytes_per_sample = 2  # int16 (matches dtype in :pymeth:`frames`)
        target_bytes = int(self._sr * clip_secs * bytes_per_sample)

        buffer = bytearray()
        vad = SilenceGate(threshold_ms=vad_threshold_ms, energy_gate=energy_gate)

        # Re-use the existing low-level frame generator to capture audio
        async for chunk in self.frames():
            buffer.extend(chunk.pcm)

            # Check for VAD trigger **after** buffering so we include silence
            if vad.update(chunk):
                # Flush any residual audio (< target_bytes) and exit
                if buffer:
                    yield AudioChunk(pcm=bytes(buffer), sample_rate=self._sr, channels=1)
                break

            # Emit one or more full-sized clips if we have collected enough
            while len(buffer) >= target_bytes:
                pcm = bytes(buffer[:target_bytes])
                yield AudioChunk(pcm=pcm, sample_rate=self._sr, channels=1)
                # Preserve any overflow for the next clip
                buffer = buffer[target_bytes:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AsyncStreamContext:
    """Async wrapper around blocking *sounddevice* context managers."""

    def __init__(self, stream_ctx, loop: asyncio.AbstractEventLoop) -> None:  # noqa: D401, ANN001
        self._ctx = stream_ctx
        self._loop = loop

    async def __aenter__(self) -> Self:  # noqa: D401
        await self._loop.run_in_executor(None, func=self._ctx.__enter__)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        await self._loop.run_in_executor(None, self._ctx.__exit__, exc_type, exc, tb) 