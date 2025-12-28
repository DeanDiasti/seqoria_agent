"""Basic end-to-end queue test for SpeechPipeline & SpeechClient (Phase-5)."""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from typing import AsyncIterator

import pytest

from seqoria_agent.client.speech_client.audio.sink import NullSink
from seqoria_agent.client.speech_client.audio.source import FileSource
from seqoria_agent.client.speech_client.orchestrator import SpeechClient
from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter
from seqoria_agent.models.llm_models import ChatChunk, AssistantMessage

# ---------------------------------------------------------------------------
# Dummy adapters – lightweight, no external deps
# ---------------------------------------------------------------------------


class DummySTT(BaseSTTAdapter):
    async def stream(self, frames: AsyncIterator[AudioChunk], *, language=None):
        async for _ in frames:
            # emit a single transcript then end
            yield TranscriptChunk(text="hi", is_final=True, start_ts=0, end_ts=0)
            break


class DummyTTS(BaseTTSAdapter):
    async def stream(self, text_iter: AsyncIterator[str], *, voice=None):
        async for _ in text_iter:
            # yield one silent audio chunk
            yield AudioChunk(pcm=b"\x00\x00" * 100, sample_rate=16000)
            break

    async def generate(self, text: str, *, voice: str | None = None):
        """Return silent PCM bytes."""
        return b""


# ---------------------------------------------------------------------------
# WAV helper
# ---------------------------------------------------------------------------


autouse = pytest.fixture(scope="module")

def _wav(tmp_path_factory):  # noqa: D401
    p = tmp_path_factory.mktemp("pipeline") / "src.wav"
    with wave.open(p, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x01" * 16000)  # 0.5s silence
    return p


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_basic(tmp_path_factory):  # noqa: D401
    src = FileSource(_wav(tmp_path_factory), chunk_ms=40)
    stt = DummySTT()
    tts = DummyTTS()
    sink = NullSink()

    class _FakeAI:  # noqa: D401 minimal AIClient stub
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_) -> None:  # noqa: D401
            pass

        async def stream_chat(self, user_query, history):  # noqa: D401
            yield ChatChunk(role="assistant", content="Hi", model=None, usage=None)
            yield AssistantMessage(role="assistant", content="Hi", model=None, usage=None)

    ai = _FakeAI()

    async with SpeechClient(ai_client=ai, source=src, stt=stt, tts=tts, sink=sink) as sp:
        await asyncio.wait_for(sp.converse(), timeout=3) 