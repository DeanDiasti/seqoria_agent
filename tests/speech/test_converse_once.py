from __future__ import annotations

import asyncio, io, wave
from typing import AsyncIterator

import pytest

from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink
from seqoria_agent.client.speech_client.audio.source import BaseAudioSource
from seqoria_agent.client.speech_client.orchestrator import SpeechClient
from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter

# ---------------------------------------------------------------------------
# Dummy adapters for batch path
# ---------------------------------------------------------------------------


class DummySink(BaseAudioSink):
    def __init__(self):
        self.calls = 0

    async def consume(self, chunk: AudioChunk):  # noqa: D401
        self.calls += 1


class DummySTT(BaseSTTAdapter):
    async def transcribe(self, pcm, sample_rate, channels, *, language=None):  # noqa: D401
        # Return one transcript chunk regardless of input
        return [TranscriptChunk(text="hello world", is_final=True, start_ts=0, end_ts=1)]

    async def stream(self, frames: AsyncIterator[AudioChunk], *, language=None):  # noqa: D401
        # Not used in this test – yield nothing
        if False:
            yield  # pragma: no cover


class DummyTTS(BaseTTSAdapter):
    async def generate(self, text: str, *, voice: str | None = None):  # noqa: D401
        # Return 20 ms of silence @16 kHz mono PCM
        return b"\x00\x00" * 320

    async def stream(self, text_iter: AsyncIterator[str], *, voice=None):  # noqa: D401
        async for _ in text_iter:
            yield AudioChunk(pcm=b"\x00\x00", sample_rate=16000)


class FakeAIClient:  # noqa: D401 – minimal stub for _agentic_once path
    async def chat(self, user_query, history, *, max_turns=5):  # noqa: D401
        yield  # never used in this test

    async def stream_chat(self, *_, **__):  # noqa: D401
        yield  # not used


# ---------------------------------------------------------------------------
# Helper – create tiny WAV blob containing silence
# ---------------------------------------------------------------------------

def _make_wav_blob(frames: int = 160, sample_rate: int = 16000):  # 10 ms mono silence
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_converse_once_wav():  # noqa: D401
    ai = FakeAIClient()
    stt = DummySTT()
    tts = DummyTTS()
    sink = DummySink()

    wav_blob = _make_wav_blob()

    # Source and STT are unused in batch helper – provide placeholders
    class NullSource(BaseAudioSource):
        async def frames(self):
            if False:
                yield  # pragma: no cover

    src = NullSource()

    async with SpeechClient(ai_client=ai, source=src, stt=stt, tts=tts, sink=sink) as sc:
        pcm = await sc.converse_once(wav_bytes=wav_blob)

    assert pcm  # non-empty bytes returned
    assert sink.calls == 1  # sink consumed reply once 

# Module-level marker
pytestmark = pytest.mark.speech 