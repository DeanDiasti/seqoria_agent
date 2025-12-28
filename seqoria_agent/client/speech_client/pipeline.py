"""
This module wires together:
    AudioSource → STTAdapter → SpeechClient (orchestrator) → TTSAdapter → AudioSink
without yet integrating a full AI/LLM loop.  Queues are exposed so the
orchestrator can inject text into *tts_text_q* and read transcripts from
*transcript_q*.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import AsyncIterator, Callable, Optional

from seqoria_agent.client.speech_client.audio.source import BaseAudioSource
from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter
from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk
from seqoria_agent.client.speech_client.utils import SilenceGate

__all__: list[str] = ["SpeechPipeline"]


class SpeechPipeline:
    """Concurrent glue between the 4 streaming adapters.

    Parameters
    ----------
    source
        Audio input provider (mic, file, etc.).
    stt
        Speech-to-text adapter converting audio to transcripts.
    tts
        Text-to-speech adapter converting assistant text to audio.
    sink
        Audio output consumer.
    max_queue
        Optional queue size - defaults to 50.
    """

    def __init__(self,
                 *,
                 source: BaseAudioSource,
                 stt: BaseSTTAdapter,
                 tts: BaseTTSAdapter,
                 sink: BaseAudioSink,
                 vad_timeout_ms: int = 600,
                 stt_quiet_ms: int = 800,
                 event_cb: Optional[Callable[[str, dict], None]] = None,
                 max_queue: int = 50) -> None:
        self.source = source
        self.stt = stt
        self.tts = tts
        self.sink = sink

        # Incremental VAD helper (Phase-11)
        self._vad = SilenceGate(threshold_ms=vad_timeout_ms)
        self._vad_triggers: int = 0  # number of end-of-utterance detections

        # STT quiet-timer configuration (Phase-12)
        if stt_quiet_ms <= 0:
            raise ValueError("stt_quiet_ms must be positive")
        self._stt_quiet_ms = int(stt_quiet_ms)
        self._stt_timeouts: int = 0

        # optional event callback
        self._event_cb = event_cb

        self.audio_q: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=max_queue)
        self.transcript_q: asyncio.Queue[TranscriptChunk | None] = asyncio.Queue(maxsize=max_queue)
        self.tts_text_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max_queue)
        self.tts_audio_q: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=max_queue)

        self._tasks: list[asyncio.Task] = []
        self._closing = asyncio.Event()

    # ------------------------------------------------------------------
    # Public metrics
    # ------------------------------------------------------------------

    @property
    def vad_triggers(self) -> int:  # noqa: D401
        """Number of times trailing-silence exceeded *vad_timeout_ms*."""

        return self._vad_triggers

    @property
    def stt_timeouts(self) -> int:  # noqa: D401
        """Number of times STT quiet-timer closed an utterance."""

        return self._stt_timeouts

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def start(self) -> None:  # noqa: D401
        """Create background tasks and start processing."""

        self._tasks = [asyncio.create_task(self._pump_source(), name="speech_src"),
                       asyncio.create_task(self._pump_stt(), name="speech_stt"),
                       asyncio.create_task(self._pump_tts(), name="speech_tts"),
                       asyncio.create_task(self._pump_sink(), name="speech_sink")]

    async def stop(self) -> None:  # noqa: D401
        """Cancel all background tasks and drain queues."""
        self._closing.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with suppress(asyncio.CancelledError):
                await t
        self._tasks.clear()

        # Allow adapters/sink to close gracefully if they expose .close()
        for obj in (self.source, self.stt, self.tts, self.sink):
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                with suppress(Exception):
                    await asyncio.wait_for(close_fn(), timeout=1)  # type: ignore[arg-type]

    # Internal pump workers ---------------------------------------------------

    async def _pump_source(self) -> None:  # noqa: D401
        speaking = False  # whether we are currently inside a user utterance

        try:
            async for chunk in self.source.frames():
                # Determine if current frame is considered silent according to gate
                rms_db = self._vad._rms_db(chunk.pcm)  # type: ignore[attr-defined]
                is_silent = rms_db <= -self._vad._gate  # type: ignore[attr-defined]

                if not speaking:
                    # Skip leading silence until we hit first speech frame
                    if is_silent:
                        self._vad.update(chunk)  # still feed gate for initial state
                        continue

                    # First non-silent frame – mark speech start
                    speaking = True
                    self._vad.reset()
                    await self.audio_q.put(chunk)
                    continue  # start accumulating after first frame

                # We are in an active speech segment ----------------------

                # Feed gate regardless to track trailing silence
                trailing = self._vad.update(chunk)

                # Forward *only* non-silent frames to STT
                if not is_silent:
                    await self.audio_q.put(chunk)

                # If trailing silence budget met → close current utterance
                if trailing:
                    self._vad_triggers += 1
                    speaking = False
                    await self.audio_q.put(None)
                    self._vad.reset()
                    if self._event_cb:
                        self._event_cb("vad_trigger", {"timestamp": time.time()})

        finally:
            # Ensure downstream components terminate
            await self.audio_q.put(None)  # final sentinel – closes STT side

    async def _audio_iter(self) -> AsyncIterator[AudioChunk]:  # noqa: D401
        while True:
            chunk = await self.audio_q.get()
            if chunk is None:
                break
            yield chunk

    async def _pump_stt(self) -> None:  # noqa: D401
        last_text_ts = time.monotonic()  # initialise timer at coroutine start

        try:
            try:
                stream_obj = self.stt.stream(frames=self._audio_iter())
            except TypeError:
                # Fallback for adapters whose *stream* expects the frames
                # iterator as a positional argument (legacy signature).
                stream_obj = self.stt.stream(self._audio_iter())

            # Allow adapters to return either an *async iterator* **or** an
            # awaitable resolving to an iterable (the dummy adapters used in
            # the test-suite take the latter approach).
            if hasattr(stream_obj, "__aiter__"):
                transcripts_iter: AsyncIterator[TranscriptChunk] = stream_obj  # type: ignore[assignment]
            else:
                result = await stream_obj  # type: ignore[func-returns-value]

                async def _iter_from_sync(seq):  # type: ignore[misc]
                    for item in seq or []:
                        yield item

                transcripts_iter = _iter_from_sync(result)

            async for transcript in transcripts_iter:
                # Reset timer only when transcript carries non-empty text.
                if transcript.text.strip():
                    last_text_ts = time.monotonic()

                await self.transcript_q.put(transcript)

                # Quiet-timer check – allow a grace period after last token.
                if time.monotonic() - last_text_ts >= self._stt_quiet_ms / 1000.0:
                    self._stt_timeouts += 1
                    if self._event_cb:
                        self._event_cb("stt_timeout", {"timestamp": time.time()})
                    break

        finally:
            # If the iterator completed without triggering the quiet timer, give it one
            # last chance based on the elapsed time since *last_text_ts*.
            elapsed = time.monotonic() - last_text_ts
            remaining = (self._stt_quiet_ms / 1000.0) - elapsed
            if remaining > 0:
                try:
                    await asyncio.sleep(remaining)
                except asyncio.CancelledError:
                    pass

            if elapsed >= self._stt_quiet_ms / 1000.0:
                self._stt_timeouts += 1
                if self._event_cb:
                    self._event_cb("stt_timeout", {"timestamp": time.time()})

            # Ensure a sentinel is always forwarded
            await self.transcript_q.put(None)

    async def _text_iter(self) -> AsyncIterator[str]:  # noqa: D401
        while True:
            txt = await self.tts_text_q.get()
            if txt is None:
                break
            yield txt

    async def _pump_tts(self) -> None:  # noqa: D401
        """Continuously convert assistant text → audio until pipeline closes."""

        while not self._closing.is_set():
            try:
                # Start (or restart) a TTS stream for the next utterance.
                if hasattr(self.tts, "pseudo_stream"):
                    tts_stream_obj = self.tts.pseudo_stream(text_iter=self._text_iter())  # type: ignore[attr-defined]
                else:
                    tts_stream_obj = self.tts.stream(text_iter=self._text_iter())

                if hasattr(tts_stream_obj, "__aiter__"):
                    audio_iter: AsyncIterator[AudioChunk] = tts_stream_obj  # type: ignore[assignment]
                else:
                    result = await tts_stream_obj  # type: ignore[func-returns-value]

                    async def _iter_from_sync(seq):  # type: ignore[misc]
                        for item in seq or []:
                            yield item

                    audio_iter = _iter_from_sync(seq=result)

                async for chunk in audio_iter:
                    await self.tts_audio_q.put(item=chunk)

            except asyncio.CancelledError:
                break

            # Loop back to listen for the next utterance. If *tts_text_q*
            # receives a *None*, *_text_iter()* returns and the inner
            # *async for* completes, allowing us to restart TTS while keeping
            # the same background task alive.

        # Pipeline shutting down – flush sentinel to sink.
        await self.tts_audio_q.put(item=None)

    async def _pump_sink(self) -> None:  # noqa: D401
        while True:
            chunk = await self.tts_audio_q.get()
            if chunk is None:
                break
            await self.sink.consume(chunk)

    # ------------------------------------------------------------------
    # Convenience iterator helpers
    # ------------------------------------------------------------------

    async def transcript_q_iter(self):  # noqa: D401
        """Async generator yielding :class:`TranscriptChunk` items until sentinel.

        Exposes the internal *transcript_q* queue to callers that want to
        consume recognised speech fragments directly (e.g. higher-level
        conversation loops).  The helper hides the *None* sentinel used
        internally to mark end-of-stream.
        """

        while True:
            t = await self.transcript_q.get()
            if t is None:
                break
            yield t 