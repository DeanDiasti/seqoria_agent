"""Contract tests for speech pipeline adapter interfaces (Phase 2)."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter
from seqoria_agent.client.speech_client.audio.source import BaseAudioSource, NullSource
from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink, NullSink


# ---------------------------------------------------------------------------
# Dummy implementations for interface compliance
# ---------------------------------------------------------------------------

class DummySTT(BaseSTTAdapter):
    async def stream(
        self, frames: AsyncIterator[AudioChunk], *, language: str | None = None
    ):
        async for _ in frames:
            yield TranscriptChunk(text="ok", is_final=True, start_ts=0, end_ts=0)


class DummyTTS(BaseTTSAdapter):
    async def stream(
        self, text_iter: AsyncIterator[str], *, voice: str | None = None
    ):
        async for _ in text_iter:
            yield AudioChunk(pcm=b"", sample_rate=16000)

    async def generate(self, text: str, *, voice: str | None = None):
        """Return silent PCM bytes for given text."""
        return b""


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stt_contract():
    assert issubclass(DummySTT, BaseSTTAdapter)

    stt = DummySTT()

    async def _empty():  # noqa: D401
        if False:
            yield  # pragma: no cover

    # Should iterate without raising
    async for _ in stt.stream(_empty()):
        pass


@pytest.mark.asyncio
async def test_tts_contract():
    assert issubclass(DummyTTS, BaseTTSAdapter)


def test_audio_nulls():
    assert issubclass(NullSource, BaseAudioSource)
    assert issubclass(NullSink, BaseAudioSink) 