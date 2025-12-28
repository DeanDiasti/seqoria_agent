"""
`SpeechClient` owns a :class:`SpeechPipeline`, exposes an async context-manager
interface and provides a minimal ``converse()`` stub that simply speaks
"Hello world" via TTS so the pipeline wiring can be validated without an LLM.
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import Callable, Dict, Self, Any, AsyncIterator
from contextlib import suppress
import time

from seqoria_agent.client.speech_client.pipeline import SpeechPipeline
from seqoria_agent.client.speech_client.audio.source import BaseAudioSource
from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink
from seqoria_agent.client.speech_client.utils import MovingAverageTimer
from seqoria_agent.client.speech_client.utils import _SpeechMetrics
from seqoria_agent.client.ai_client.base import AIClient
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter
from seqoria_agent.models.llm_models import (ChatChunk, 
                                             AssistantMessage, 
                                             ToolCallRequestList, 
                                             ToolCallRequest, 
                                             ToolCallResult,
                                             UserMessage)
from seqoria_agent.models.speech_models import AudioChunk

class SpeechClient:
    """Facade combining adapters and the pipeline into one easy object."""

    def __init__(self,
                 *,
                 ai_client: AIClient,
                 source: BaseAudioSource,
                 stt: BaseSTTAdapter,
                 tts: BaseTTSAdapter,
                 sink: BaseAudioSink,
                 vad_timeout_ms: int = 600,
                 stt_quiet_ms: int = 1500) -> None:
        self.ai = ai_client
        # Wrapper to capture pipeline events and forward via _fire while updating metrics
        def _on_pipeline_event(evt: str, payload: dict) -> None:  # noqa: D401
            if evt == "vad_trigger":
                self._metrics.vad_events += 1
            elif evt == "stt_timeout":
                self._metrics.stt_timeouts += 1
            self._fire(event=evt, payload=payload)

        self._pipeline = SpeechPipeline(source=source,
                                        stt=stt,
                                        tts=tts,
                                        sink=sink,
                                        vad_timeout_ms=vad_timeout_ms,
                                        stt_quiet_ms=stt_quiet_ms,
                                        event_cb=_on_pipeline_event)
        self._events: Dict[str, list[Callable]] = {}
        self._assistant_speaking = False
        self._lat_timer = MovingAverageTimer()
        self._vad_timeout_ms = vad_timeout_ms
        self._history: list[dict] = []
        self._ai_task: asyncio.Task | None = None
        self._metrics = _SpeechMetrics(latency=self._lat_timer)
        self.metrics = self._metrics  # public alias
        self._timestamp_last_user: float | None = None

    # Event subscription API (lightweight – non-blocking callbacks) ------------

    def on(self, event: str, fn: Callable) -> None:  # noqa: D401
        """Register *fn* to be called when *event* fires (Phase-6 expansion)."""
        self._events.setdefault(event, []).append(fn)

    @property
    def events(self) -> MappingProxyType[Any, Any]:  # noqa: D401 – expose read-only mapping
        return MappingProxyType(self._events)

    # Async context-manager ----------------------------------------------------

    async def __aenter__(self) -> Self:  # noqa: D401
        await self._pipeline.start()
        # Background helper that *drives* the stream_converse generator
        async def _driver() -> None:  # noqa: D401
            async for _ in self.stream_converse():
                pass  # discard yielded parts unless caller consumes directly
        # Launch the driver so existing .converse() keeps working.
        self._ai_task = asyncio.create_task(_driver(), name="speech_stream_converse")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        await self._pipeline.stop()
        if self._ai_task:
            self._ai_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._ai_task

    # Public API --------------------------------------------------------------

    async def converse(self) -> None:  # noqa: D401
        """Run conversation until audio source ends"""
        await self._ai_task  # Wait until AI loop completes (source exhausted)

    # ------------------------------------------------------------------
    # Low-level streaming primitives (public) ---------------------------
    # ------------------------------------------------------------------

    async def iter_audio_chunks(self):  # noqa: D401
        """Yield raw :class:`AudioChunk` objects from the underlying source.

        This is a thin wrapper that simply exposes the internal audio frames
        generator so advanced callers can assemble fully-custom pipelines
        without touching private queues.
        """
        # NOTE: SpeechPipeline._audio_iter is an internal helper but stable.
        async for chunk in self._pipeline._audio_iter():
            yield chunk

    async def iter_transcripts(self):  # noqa: D401
        """Yield :class:`TranscriptChunk` objects from the STT adapter."""
        async for t in self._pipeline.transcript_q_iter():
            yield t

    async def iter_utterances(self, 
                              *, 
                              timeout_ms: int | None = None, 
                              quiet_ms: int | None = None):  # noqa: D401,E501
        """Convert live transcript fragments into *complete* user utterances.

        Parameters
        ----------
        timeout_ms
            Optional override for VAD trailing-silence budget (defaults to
            *vad_timeout_ms* passed to the constructor).
        quiet_ms
            Optional override for STT quiet timer (defaults to *stt_quiet_ms*
            passed to the constructor).
        Yields
        ------
        str
            The full text recognised for a single user turn.
        """
        timeout_ms = int(timeout_ms or self._vad_timeout_ms)
        quiet_ms = int(quiet_ms or self._pipeline._stt_quiet_ms)  # type: ignore[attr-defined]

        user_buf = ""
        user_started = False
        last_transcript_ts: float | None = None

        async for t in self.iter_transcripts():
            # Fire user_start only once per utterance
            if not user_started:
                user_started = True
                self._fire(event="user_start", 
                           payload={"timestamp": time.time()})

            user_buf += (" " if user_buf else "") + t.text
            self._fire(event="transcript", 
                       payload={"timestamp": time.time(), 
                                "text": t.text, 
                                "final": t.is_final})

            # Track time of last partial
            last_transcript_ts = time.monotonic()

            # Decide if utterance finished
            finished = False
            if t.is_final:
                finished = True
            elif user_buf.endswith((".", "?", "!")):
                finished = True
            elif len(user_buf.split()) >= 5:
                finished = True

            if finished:
                self._fire(event="user_stop", 
                           payload={"timestamp": time.time(), 
                                    "partial": user_buf})
                yield user_buf.strip()
                user_buf = ""
                user_started = False
                last_transcript_ts = None
                continue

            # Quiet-timer fallback – force end when no new text for *quiet_ms*
            if last_transcript_ts and (time.monotonic() - last_transcript_ts) * 1000 >= quiet_ms:
                self._fire(event="stt_timeout", 
                           payload={"timestamp": time.time()})
                yield user_buf.strip()
                user_buf = ""
                user_started = False
                last_transcript_ts = None

        # Flush trailing buffer when source ends
        if user_buf:
            yield user_buf.strip()

    async def iter_llm_parts(self, 
                             user_text: str, 
                             *, 
                             history: list[dict] | None = None, 
                             max_turns: int = 5, 
                             context: dict | None = None):  # noqa: D401,E501
        """Stream pieces of the LLM reply for *user_text*.

        This stage **does not** perform TTS - it simply yields the objects
        produced by :pymeth:`AIClient.stream_chat` so that higher levels can
        decide how to render or speak them.
        """
        # If no AI client supplied (e.g. metrics-only tests) → nothing to yield
        if self.ai is None:
            return  # graceful no-op

        # Metric – count a completed user turn
        self._metrics.turn_count += 1

        if history is None:
            history = self._history

        async for part in self.ai.stream_chat(user_query=user_text, 
                                              history=history, 
                                              max_turns=max_turns, 
                                              context=context):
            yield part

    async def iter_tts_audio(self, parts_iter):  # noqa: D401
        """Convert LLM parts into audio and yield :class:`AudioChunk`s."""

        # Accept both sync & async iterables for convenience
        if not hasattr(parts_iter, "__aiter__"):
            async def _to_async(seq):  # type: ignore[misc]
                for item in seq:
                    yield item
            parts_iter = _to_async(parts_iter)

        # Helper to queue text to TTS and emit assistant_start/stop events
        async def _send_to_tts(text: str, *, end: bool = False) -> None:
            if not text and not end:
                return
            if text and not self._assistant_speaking:
                self._assistant_speaking = True
                self._fire(event="assistant_start", 
                           payload={"timestamp": time.time(), 
                                    "preview": text[:20]})
                if self._timestamp_last_user is not None:
                    self._lat_timer.add(value=time.monotonic() - self._timestamp_last_user)
                    self._timestamp_last_user = None
            if text:
                await self._pipeline.tts_text_q.put(item=text)
            if end:
                await self._pipeline.tts_text_q.put(item=None)
                self._assistant_speaking = False
                self._fire(event="assistant_stop", payload={"timestamp": time.time()})

        sent_so_far = ""  # text already queued to TTS – used to compute deltas

        async for part in parts_iter:
            if isinstance(part, (ChatChunk, AssistantMessage)):
                full_text = part.content or ""
                # Compute *unsent* suffix to avoid repetition.
                delta = full_text[len(sent_so_far):] if full_text.startswith(sent_so_far) else full_text

                if delta:
                    await _send_to_tts(text=delta, end=False)
                    sent_so_far = full_text  # advance pointer

                if isinstance(part, AssistantMessage):
                    # Mark end of utterance for TTS
                    await _send_to_tts(text="", end=True)
                    sent_so_far = ""  # reset for next turn

            elif isinstance(part, ToolCallRequestList):
                await self._handle_tool_calls(req_list=part)
            # Yield *audio* chunks produced for this part
            while True:
                try:
                    chunk = self._pipeline.tts_audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if chunk is not None:
                    yield chunk

    # ------------------------------------------------------------------
    # High-level convenience loop --------------------------------------
    # ------------------------------------------------------------------

    async def stream_converse(self, history: list[dict] | None = None, *, max_turns: int = 5, context: dict | None = None):  # noqa: D401,E501
        """Full duplex conversation loop (async generator).

        Yields each piece received from the LLM (`ChatChunk`, `AssistantMessage`,
        `ToolCallRequestList`).  Callers may ignore the yielded values if they
        only care about audio playback.
        """
        if history is None:
            history = self._history

        async for utterance in self.iter_utterances():
            # Store timestamp for latency measurement
            self._timestamp_last_user = time.monotonic()

            # Emit user message so callers can include in UI/history
            user_msg = UserMessage(role="user", content=utterance)
            # Append to shared history *before* querying LLM
            history.append(user_msg.model_dump())

            if self.ai is None:
                # Metrics-only mode – no LLM/TTS
                continue

            parts_stream = self.iter_llm_parts(user_text=utterance,
                                               history=history,
                                               max_turns=max_turns,
                                               context=context)

            async for part in parts_stream:
                # Stream part immediately to TTS
                async for _ in self.iter_tts_audio(parts_iter=iter([part])):
                    pass
                yield part

    async def converse_once(self,
                            *,
                            audio_iter: AsyncIterator[AudioChunk] | None = None,
                            wav_bytes: bytes | None = None,
                            language: str | None = None) -> bytes:  # noqa: D401
        """Batch speech-to-speech helper - single user utterance in / single reply out.

        Parameters
        ----------
        audio_iter
            Async iterator yielding :class:`AudioChunk` objects.  Mutually
            exclusive with *wav_bytes*.
        wav_bytes
            Pre-recorded WAV payload (PCM, little-endian).  Mutually exclusive
            with *audio_iter*.
        language
            Optional BCP-47 language tag forwarded to STT adapter.

        Returns
        -------
        bytes
            Assistant reply PCM buffer (format determined by the active TTS
            adapter).
        """

        # ------------------------------------------------------------------
        # Validate input – exactly one of audio_iter | wav_bytes must be given
        # ------------------------------------------------------------------
        if (audio_iter is None) == (wav_bytes is None):  # both None *or* both set
            raise ValueError("Provide exactly one of 'audio_iter' or 'wav_bytes'.")

        # Normalise input → PCM bytes + sample_rate + channels
        pcm_bytes: bytes
        sample_rate: int
        channels: int

        if audio_iter is not None:
            pcm_collector = bytearray()
            first_sr = first_ch = None
            async for chunk in audio_iter:
                pcm_collector.extend(chunk.pcm)
                # Track SR/channels from the *first* chunk (assume constant)
                if first_sr is None:
                    first_sr = chunk.sample_rate
                    first_ch = chunk.channels
            if first_sr is None:
                # empty iterator – nothing to transcribe
                return b""

            pcm_bytes = bytes(pcm_collector)
            sample_rate = int(first_sr)
            channels = int(first_ch)

        else:  # wav_bytes path
            import io, wave  # local import to avoid unnecessary dependency at runtime

            buf = io.BytesIO(wav_bytes)  # type: ignore[arg-type]
            with wave.open(buf, "rb") as wf:  # noqa: S310 – in-memory stream
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                pcm_bytes = wf.readframes(wf.getnframes())

        # Speech-to-text – single call
        transcripts = await self._pipeline.stt.transcribe(pcm=pcm_bytes,
                                                          sample_rate=sample_rate,
                                                          channels=channels,
                                                          language=language)

        user_text = " ".join(t.text for t in transcripts) if transcripts else ""
        if not user_text:
            return b""  # nothing to say – defensive

        # LLM turn → assistant text
        assistant_text = await self._agentic_once(user_text=user_text)

        # TTS one-shot
        pcm_reply = await self._pipeline.tts.generate(text=assistant_text)

        # Attempt to infer reply SR from adapter attribute; fall back to input SR
        sr_reply = getattr(self._pipeline.tts, "sample_rate", sample_rate)

        # Optional playback via sink
        if self._pipeline.sink is not None:
            try:
                await self._pipeline.sink.consume(chunk=AudioChunk(pcm=pcm_reply, 
                                                                   sample_rate=sr_reply, 
                                                                   channels=1))
            except Exception:  # pragma: no cover – sink errors non-fatal for batch path
                pass

        return pcm_reply

    # Internal helpers --------------------------------------------------------

    async def _agentic_once(self, user_text: str) -> str:  # noqa: D401
        """Single non-streaming AI turn helper

        Parameters
        ----------
        user_text:
            The complete user utterance to send to the LLM.

        Returns
        -------
        str
            Assistant response (empty string if no textual reply).
        """

        # Track a completed turn for metrics insight.
        self._metrics.turn_count += 1

        # Delegate to the synchronous chat helper with a small turn budget.
        async for part in self.ai.chat(user_query=user_text,
                                       history=self._history,
                                       max_turns=5):
            # The chat() generator yields either an AssistantMessage or a
            # ToolCallRequestList.  We return as soon as we get the first
            # plain assistant reply.
            if isinstance(part, AssistantMessage):
                return part.content or ""

        # Defensive – in unusual cases the loop could finish without an
        # AssistantMessage (e.g. exhausted turn budget).  Return empty so
        # callers can decide how to proceed.
        return ""

    # Event forwarding --------------------------------------------------------

    def _fire(self, event: str, payload: dict) -> None:  # noqa: D401, ANN001
        for fn in self._events.get(event, []):
            try:
                asyncio.create_task(fn(payload))
            except Exception:
                continue

    # Tool-call handling -----------------------------------------------------

    async def _handle_tool_calls(self, req_list: ToolCallRequestList) -> None:  # noqa: D401
        # Stop current speech
        if self._assistant_speaking:
            await self._pipeline.tts_text_q.put(item=None)
            self._assistant_speaking = False
            self._fire(event="assistant_stop", payload={"timestamp": time.time()})

        self._metrics.tool_calls += len(req_list.tool_calls)

        # Execute tools via underlying executor
        results: list[ToolCallResult] = await self.ai.executor.run_calls(tool_calls=req_list)

        for res in results:
            text_out = ""
            if hasattr(res, "error") and res.error:
                # Failed tool – speak short apology
                text_out = "Sorry, the tool failed: " + res.error[:80]
            else:
                raw = getattr(res, "result", "")
                if isinstance(raw, str) and len(raw) > 300:
                    # summarise via meta-tool
                    summary_req = ToolCallRequest(id="summarize", name="__summarize__", arguments={"text": raw})
                    summary_res = (await self.ai.executor.run_calls(tool_calls=[summary_req]))[0]
                    text_out = getattr(summary_res, "result", "")
                else:
                    text_out = str(object=raw)

            await self._speak_assistant(text=text_out)

    async def _speak_assistant(self, text: str) -> None:  # noqa: D401
        """Queue *text* to TTS handling start/stop events & latency timer."""
        if not text:
            return
        await self._send_to_tts(text)
        await self._pipeline.tts_text_q.put(None)  # mark end
        self._assistant_speaking = False
        self._fire(event="assistant_stop", payload={"timestamp": time.time()}) 