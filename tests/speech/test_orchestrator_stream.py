"""Stream integration test for SpeechClient (Phase-6)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from seqoria_agent.client.speech_client.audio.sink import NullSink
from seqoria_agent.client.speech_client.audio.source import BaseAudioSource
from seqoria_agent.client.speech_client.orchestrator import SpeechClient
from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter
from seqoria_agent.models.llm_models import ChatChunk, AssistantMessage

# ---------------------------------------------------------------------------
# Dummy adapters
# ---------------------------------------------------------------------------


# Dummy source yields one silent chunk ---------------------------------------


class DummySource(BaseAudioSource):
    async def frames(self):  # noqa: D401
        yield AudioChunk(pcm=b"\x00\x00" * 100, sample_rate=16000)


class DummySTT(BaseSTTAdapter):
    async def stream(self, frames: AsyncIterator[AudioChunk], *, language=None):
        # emit one final transcript then stop
        yield TranscriptChunk(text="hi there.", is_final=True, start_ts=0, end_ts=1)


class DummyTTS(BaseTTSAdapter):
    def __init__(self):
        self.outputs: list[str] = []

    async def stream(self, text_iter: AsyncIterator[str], *, voice=None):
        async for txt in text_iter:
            self.outputs.append(txt)
            yield AudioChunk(pcm=b"\x00\x00" * 100, sample_rate=16000)
            break

    async def generate(self, text: str, *, voice: str | None = None):
        """Return silent PCM bytes for given text."""
        return b""


class FakeAIClient:  # noqa: D401 – minimal stub
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_) -> None:  # noqa: D401
        pass

    async def stream_chat(self, user_query, history):  # noqa: D401
        yield ChatChunk(role="assistant", content="Hi 👋", model=None, usage=None)
        yield AssistantMessage(role="assistant", content="Hi 👋", model=None, usage=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_stream_callbacks():  # noqa: D401
    ai = FakeAIClient()
    stt = DummySTT()
    tts = DummyTTS()

    src = DummySource()
    sink = NullSink()

    events_fired = {}

    def _make_cb(name):  # noqa: D401
        def _cb(payload):
            events_fired[name] = True
        return _cb

    async with SpeechClient(ai_client=ai, source=src, stt=stt, tts=tts, sink=sink) as sc:
        for ev in ("user_start", "user_stop", "assistant_start", "assistant_stop"):
            sc.on(ev, _make_cb(ev))
        await asyncio.wait_for(sc.converse(), timeout=3)

    # Ensure assistant text reached TTS
    assert "Hi" in " ".join(tts.outputs)
    # Basic callback sanity
    assert events_fired["assistant_start"] and events_fired["assistant_stop"]


# ---------------------------------------------------------------------------
# New primitives – iter_llm_parts validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter_llm_parts_primitive():  # noqa: D401
    ai = FakeAIClient()
    stt = DummySTT()
    tts = DummyTTS()

    # Pipeline not required for this primitive; supply NullSource / NullSink
    src = DummySource()
    sink = NullSink()

    sc = SpeechClient(ai_client=ai, source=src, stt=stt, tts=tts, sink=sink)

    parts = []
    async for part in sc.iter_llm_parts("hello", history=[]):
        parts.append(part)

    # Should contain both streaming chunk and final assistant message
    assert any(isinstance(p, ChatChunk) for p in parts)
    assert any(isinstance(p, AssistantMessage) for p in parts) 