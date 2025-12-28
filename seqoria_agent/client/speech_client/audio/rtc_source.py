"""WebRTC MediaStreamTrack → AudioChunk source."""

from __future__ import annotations

from typing import AsyncIterator

from seqoria_agent.models.speech_models import AudioChunk
from seqoria_agent.client.speech_client.audio.source import BaseAudioSource


class RTCSource(BaseAudioSource):
    """Adapt an aiortc MediaStreamTrack into the AudioSource interface."""

    def __init__(self, track, *, chunk_ms: int = 20) -> None:  # type: ignore[no-any-unbound]
        try:
            from aiortc import MediaStreamTrack  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover – optional dep
            raise ImportError(
                "RTCSource requires 'aiortc' - install via 'pip install \"seqoria-agent[speech]\"'"
            ) from exc

        if not isinstance(track, MediaStreamTrack):
            raise TypeError("track must be an aiortc.MediaStreamTrack")
        self._track = track
        self._chunk_ms = int(chunk_ms)
        self._buf: bytearray = bytearray()

    async def frames(self) -> AsyncIterator[AudioChunk]:  # noqa: D401
        # Lazy import to avoid mandatory aiortc in environments that don't need RTC.
        from aiortc.contrib.media import encode  # type: ignore

        sample_rate = 48000  # WebRTC default – will adjust if frame reports differently
        channels = 1

        while True:
            frame = await self._track.recv()
            if frame is None:
                break
            sample_rate = getattr(frame, "sample_rate", sample_rate)
            pcm_bytes = frame.to_ndarray().tobytes()  # type: ignore[attr-defined]
            self._buf.extend(pcm_bytes)

            bytes_per_ms = sample_rate * channels * 2 / 1000
            target_bytes = int(bytes_per_ms * self._chunk_ms)

            while len(self._buf) >= target_bytes > 0:
                chunk_bytes = self._buf[:target_bytes]
                del self._buf[:target_bytes]
                yield AudioChunk(pcm=bytes(chunk_bytes), sample_rate=sample_rate, channels=channels) 