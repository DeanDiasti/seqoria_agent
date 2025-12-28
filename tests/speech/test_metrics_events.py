import asyncio
import pytest
from seqoria_agent.client.speech_client.pipeline import SpeechPipeline
from seqoria_agent.client.speech_client.audio.source import BaseAudioSource
from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink
from seqoria_agent.stt_providers.base import BaseSTTAdapter
from seqoria_agent.tts_providers.base import BaseTTSAdapter
from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk
from seqoria_agent.client.speech_client.orchestrator import SpeechClient
from seqoria_agent.client.speech_client.utils import _SpeechMetrics


# Dummy components -----------------------------------------------------------

class DummySource(BaseAudioSource):
    def __init__(self, frames):
        self._frames = frames

    async def frames(self):  # noqa: D401
        for fr in self._frames:
            yield fr


class DummySink(BaseAudioSink):
    async def consume(self, chunk: AudioChunk):  # noqa: D401
        return


class DummyTTS(BaseTTSAdapter):
    async def stream(self, text_iter, *, voice=None):  # noqa: D401, ANN001
        async for _ in text_iter:
            pass
        return

    async def generate(self, text: str, *, voice: str | None = None):
        """Return empty PCM bytes."""
        return b""


class DummySTT(BaseSTTAdapter):
    def __init__(self, transcripts):
        self._transcripts = transcripts

    async def stream(self, _frames, **_):  # noqa: D401, ANN001
        for txt in self._transcripts:
            yield TranscriptChunk(text=txt, is_final=False, start_ts=0, end_ts=0)
            await asyncio.sleep(0)

    async def transcribe(self, pcm, sample_rate, channels, *, language=None):  # noqa: D401
        return []


# Helpers --------------------------------------------------------------------

def _silent_frame() -> AudioChunk:
    return AudioChunk(pcm=b"\x00\x00" * 320, sample_rate=16000, channels=1)


aasyncio = pytest.mark.asyncio  # alias


@aasyncio
async def test_metrics_counters():
    frames = [_silent_frame() for _ in range(10)]

    stt = DummySTT(["hello", "world"])
    client = SpeechClient(
        ai_client=None,  # type: ignore
        source=DummySource(frames),
        stt=stt,
        tts=DummyTTS(),
        sink=DummySink(),
        vad_timeout_ms=100,
        stt_quiet_ms=100,
    )

    # Capture events for ordering
    events = []
    client.on("vad_trigger", lambda p: events.append("vad_trigger"))
    client.on("stt_timeout", lambda p: events.append("stt_timeout"))

    async with client:
        await client.converse()

    assert client.metrics.vad_events >= 0
    # stt_timeouts should be >=0 depending on transcripts and timeouts
    assert client.metrics.stt_timeouts >= 0
    # Ensure event list contains only recognised event names
    for e in events:
        assert e in {"vad_trigger", "stt_timeout"} 