import sys
import types
import asyncio

import pytest

# ---------------------------------------------------------------------------
# Ensure the optional `ffmpeg` dependency is importable during tests.
# If the real package is not available in the environment we inject a stub so
# that `seqoria_agent.stt_providers.fireworks_stt` can be imported safely.
# ---------------------------------------------------------------------------
if "ffmpeg" not in sys.modules:
    stub = types.ModuleType("ffmpeg")

    def _input(*_args, **_kwargs):  # type: ignore[override]
        class _Dummy:
            def output(self, *args, **kwargs):  # noqa: D401, ANN001
                return self

            def run_async(self, *args, **kwargs):  # noqa: D401, ANN001
                class _Proc:  # minimal stdout behaviour
                    def __init__(self):
                        import io
                        self.stdout = io.BytesIO(b"\x00\x00" * 4096)
                        self.stderr = io.BytesIO()

                    def kill(self):
                        pass

                    def wait(self):
                        pass

                    def communicate(self):  # noqa: D401, ANN001
                        return self.stdout.read(), self.stderr.read()

                return _Proc()

        return _Dummy()

    stub.input = _input  # type: ignore[attr-defined]
    sys.modules["ffmpeg"] = stub

# The import must happen *after* the stub insertion
from seqoria_agent.stt_providers.fireworks_stt import FireworksSTT, AudioChunk, TranscriptChunk  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _dummy_transcribe(self, *_, **__) -> list[TranscriptChunk]:  # noqa: D401, ANN001
    """Return a single fake transcript chunk regardless of input."""
    return [TranscriptChunk(text="hello", is_final=True, start_ts=0.0, end_ts=0.1)]

async def _dummy_frames():  # noqa: D401, ANN001
    yield AudioChunk(pcm=b"\x00\x00" * 3200, sample_rate=16_000, channels=1)

# ---------------------------------------------------------------------------
# STREAM TESTS
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_frames(monkeypatch):
    """Streaming with explicit frames iterator succeeds and yields transcripts."""
    monkeypatch.setattr(FireworksSTT, "_transcribe", _dummy_transcribe)

    stt = FireworksSTT(api_key="dummy", chunk_duration=100, overlap=0)

    results = [chunk async for chunk in stt.stream(frames=_dummy_frames())]
    assert results and results[0].text == "hello"


@pytest.mark.asyncio
async def test_stream_url(monkeypatch):
    """Streaming with URL delegates to _frames_from_source once."""
    monkeypatch.setattr(FireworksSTT, "_transcribe", _dummy_transcribe)

    called = {}

    def _fake_frames_from_source(self, src, *, frame_sec):  # type: ignore[override]
        called["src"] = src
        return _dummy_frames()

    monkeypatch.setattr(FireworksSTT, "_frames_from_source", _fake_frames_from_source)

    stt = FireworksSTT(api_key="dummy", chunk_duration=100, overlap=0)
    results = [chunk async for chunk in stt.stream(url="http://example.com/stream.mp3")]

    assert called["src"].startswith("http://example.com")
    assert results and results[0].text == "hello"


@pytest.mark.asyncio
async def test_stream_validation_errors(monkeypatch):
    stt = FireworksSTT(api_key="dummy")

    # None provided
    with pytest.raises(ValueError):
        async for _ in stt.stream():
            pass

    # Multiple provided
    with pytest.raises(ValueError):
        async for _ in stt.stream(frames=_dummy_frames(), url="http://x"):
            pass


# ---------------------------------------------------------------------------
# TRANSCRIBE TESTS
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transcribe_pcm(monkeypatch):
    monkeypatch.setattr(FireworksSTT, "_transcribe", _dummy_transcribe)

    stt = FireworksSTT(api_key="dummy")
    res = await stt.transcribe(pcm=b"\x00\x00" * 3200)
    assert res and res[0].text == "hello"


@pytest.mark.asyncio
async def test_transcribe_url(monkeypatch):
    monkeypatch.setattr(FireworksSTT, "_transcribe", _dummy_transcribe)

    decoded = {}

    async def _fake_decode(self, src, *, sample_rate, channels):  # noqa: D401, ANN001
        decoded["src"] = src
        return b"\x00\x00" * 3200

    monkeypatch.setattr(FireworksSTT, "_decode_source_to_pcm", _fake_decode)

    stt = FireworksSTT(api_key="dummy")
    res = await stt.transcribe(url="http://example.com/audio.mp3")

    assert decoded["src"].startswith("http://example.com")
    assert res and res[0].text == "hello"


@pytest.mark.asyncio
async def test_transcribe_validation_errors(monkeypatch):
    stt = FireworksSTT(api_key="dummy")

    with pytest.raises(ValueError):
        await stt.transcribe()

    with pytest.raises(ValueError):
        await stt.transcribe(pcm=b"a", file="x.wav") 