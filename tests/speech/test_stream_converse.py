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

# Mark entire module for selective CI runs
pytestmark = pytest.mark.speech

# ---------------------------------------------------------------------------
# Dummy adapters for test
# ---------------------------------------------------------------------------


class DummySource(BaseAudioSource):
    async def frames(self):  # noqa: D401
        # Yield one silent frame – enough to satisfy pipeline.
        yield AudioChunk(pcm=b"\x00\x00" * 100, sample_rate=16000, channels=1)


class DummySTT(BaseSTTAdapter):
    async def stream(self, frames: AsyncIterator[AudioChunk], *, language=None):  # noqa: D401
        # Emit two complete user utterances.
        yield TranscriptChunk(text="hello.", is_final=True, start_ts=0, end_ts=1)
        yield TranscriptChunk(text="how are you?", is_final=True, start_ts=1, end_ts=2)


class DummyTTS(BaseTTSAdapter):
    def __init__(self):
        self.outputs: list[str] = []

    async def stream(self, text_iter: AsyncIterator[str], *, voice=None):  # noqa: D401
        async for txt in text_iter:
            self.outputs.append(txt)
            yield AudioChunk(pcm=b"\x00\x00" * 100, sample_rate=16000, channels=1)

    async def generate(self, text: str, *, voice: str | None = None):
        return b""


class FakeAIClient:  # noqa: D401 – minimal stub
    def __init__(self):
        self.call_idx = 0
        self.executor = type("_DummyExec", (), {"run_calls": staticmethod(lambda tool_calls: [])})

    async def stream_chat(self, user_query, history):  # noqa: D401
        self.call_idx += 1
        yield ChatChunk(role="assistant", content=f"reply {self.call_idx} partial", model=None, usage=None)
        yield AssistantMessage(role="assistant", content=f"reply {self.call_idx} full", model=None, usage=None)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_converse_multiple_turns():  # noqa: D401
    ai = FakeAIClient()
    stt = DummySTT()
    tts = DummyTTS()
    src = DummySource()
    sink = NullSink()

    async with SpeechClient(ai_client=ai, source=src, stt=stt, tts=tts, sink=sink) as sc:
        await asyncio.wait_for(sc.converse(), timeout=3)

    # Expect two independent assistant replies captured by DummyTTS
    assert any("reply 1" in o for o in tts.outputs)
    assert any("reply 2" in o for o in tts.outputs) 