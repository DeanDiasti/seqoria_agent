import asyncio
import pytest

from seqoria_agent.client.speech_client.pipeline import SpeechPipeline
from seqoria_agent.client.speech_client.audio.source import BaseAudioSource
from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter
from seqoria_agent.models.speech_models import AudioChunk

_SAMPLE_RATE = 16_000  # 16-kHz mono
_FRAME_MS = 20  # 20-ms frames


def _silent_chunk() -> AudioChunk:
    samples = int(_SAMPLE_RATE * _FRAME_MS / 1000)
    pcm = bytes(samples * 2)  # int16 zeros
    return AudioChunk(pcm=pcm, sample_rate=_SAMPLE_RATE, channels=1)


def _speech_chunk(amplitude: int = 5_000) -> AudioChunk:
    samples = int(_SAMPLE_RATE * _FRAME_MS / 1000)
    sample_bytes = amplitude.to_bytes(2, "little", signed=True)
    pcm = sample_bytes * samples
    return AudioChunk(pcm=pcm, sample_rate=_SAMPLE_RATE, channels=1)


# ---------------------------------------------------------------------------
# Dummy components
# ---------------------------------------------------------------------------


class DummySource(BaseAudioSource):
    def __init__(self, frames):
        self._frames = frames

    async def frames(self):  # noqa: D401
        for fr in self._frames:
            yield fr


class DummySTT(BaseSTTAdapter):
    def __init__(self):
        self.frames_received: list[AudioChunk] = []

    async def stream(self, frames_iter, **_):  # noqa: D401, ANN001
        async for ch in frames_iter:
            self.frames_received.append(ch)
        return  # no transcripts

    async def transcribe(self, pcm, sample_rate, channels, *, language=None):  # noqa: D401, ANN001
        return []


class DummyTTS(BaseTTSAdapter):
    async def stream(self, text_iter, *, voice=None):  # noqa: D401, ANN001
        async for _ in text_iter:
            pass
        return

    async def generate(self, text: str, *, voice: str | None = None):
        """Return empty PCM bytes."""
        return b""


class DummySink(BaseAudioSink):
    async def consume(self, chunk: AudioChunk):  # noqa: D401
        return


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_vad_gating():
    """Only speech frames should reach STT before first sentinel."""

    frames = (
        [_silent_chunk() for _ in range(5)]  # 100 ms leading silence
        + [_speech_chunk() for _ in range(2)]  # speech
        + [_silent_chunk() for _ in range(30)]  # 600 ms silence → trigger
    )

    stt = DummySTT()
    pipeline = SpeechPipeline(
        source=DummySource(frames),
        stt=stt,
        tts=DummyTTS(),
        sink=DummySink(),
    )

    await pipeline.start()
    # Wait for pump_source task to complete (index 0)
    await pipeline._tasks[0]
    # Give other tasks chance to exhaust
    await asyncio.sleep(0)
    await pipeline.stop()

    # Expect exactly 2 speech frames forwarded
    assert len(stt.frames_received) == 2


@pytest.mark.asyncio
async def test_multiple_utterances_count():
    """Pipeline should detect and count multiple utterances."""

    frames = (
        [_silent_chunk() for _ in range(5)]
        + [_speech_chunk() for _ in range(2)]
        + [_silent_chunk() for _ in range(30)]  # first utterance ends
        + [_silent_chunk() for _ in range(10)]  # extra leading silence
        + [_speech_chunk() for _ in range(3)]  # second utterance
        + [_silent_chunk() for _ in range(30)]
    )

    pipeline = SpeechPipeline(
        source=DummySource(frames),
        stt=DummySTT(),
        tts=DummyTTS(),
        sink=DummySink(),
    )

    await pipeline.start()
    await pipeline._tasks[0]
    await asyncio.sleep(0)
    await pipeline.stop()

    assert pipeline.vad_triggers == 2, "Expected two utterance boundaries detected" 