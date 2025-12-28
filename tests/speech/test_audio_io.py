"""Tests for Audio I/O layer (Phase 4)."""

from __future__ import annotations

import asyncio
import builtins
import wave

import pytest

from seqoria_agent.client.speech_client.audio.source import FileSource, MicSource
from seqoria_agent.client.speech_client.audio.sink import SpeakerSink

# ---------------------------------------------------------------------------
# Helpers – temporary wav fixture
# ---------------------------------------------------------------------------


autouse = pytest.fixture(scope="module")

def _tmp_wav(tmp_path_factory):  # noqa: D401
    p = tmp_path_factory.mktemp("audio") / "test.wav"
    sr = 16000
    duration_sec = 1
    frames = sr * duration_sec
    with wave.open(p, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * frames)
    return p


# ---------------------------------------------------------------------------
# FileSource tests (no optional deps)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_source(tmp_path_factory):  # noqa: D401
    wav_path = _tmp_wav(tmp_path_factory)
    src = FileSource(wav_path, chunk_ms=100)

    total_bytes = 0
    chunks = 0
    async for ch in src.frames():
        chunks += 1
        total_bytes += len(ch.pcm)
        assert ch.sample_rate == 16000
        assert ch.channels == 1

    # 1 second @16 kHz, 16-bit mono → 32_000 bytes
    assert total_bytes == 16000 * 2 * 1
    assert chunks > 0


# ---------------------------------------------------------------------------
# Sounddevice dependency guards
# ---------------------------------------------------------------------------


def _patch_import_failure(monkeypatch):  # noqa: D401
    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # noqa: D401
        if name == "sounddevice":
            raise ModuleNotFoundError("No sounddevice")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


@pytest.mark.asyncio
async def test_sounddevice_missing_guard(monkeypatch):  # noqa: D401
    _patch_import_failure(monkeypatch)
    with pytest.raises(ImportError):
        MicSource()
    with pytest.raises(ImportError):
        SpeakerSink()


# ---------------------------------------------------------------------------
# Mocked sounddevice flow – requires importorskip to pass when dep missing
# ---------------------------------------------------------------------------


sd = pytest.importorskip(
    "sounddevice", reason="sounddevice not installed – skipping mocked flow test"
)


class _MockInputStream:  # noqa: D401 – minimal mock
    def __init__(self, *, channels, samplerate, blocksize, dtype, callback):
        self._callback = callback
        self._blocksize = blocksize
        self._running = False

    def __enter__(self):  # noqa: D401
        self._running = True
        # Simulate one callback invocation with silence
        self._callback(b"\x00\x00" * self._blocksize, None, None, None)
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        self._running = False
        return False


class _MockOutputStream:  # noqa: D401 – minimal mock
    def __init__(self, *, channels, samplerate, dtype):
        self.written: list[bytes] = []

    def write(self, data):  # noqa: D401
        self.written.append(data)

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False


@pytest.mark.asyncio
async def test_mocked_mic_to_speaker(monkeypatch):  # noqa: D401
    # Patch sounddevice streams
    monkeypatch.setattr(sd, "InputStream", _MockInputStream, raising=False)
    monkeypatch.setattr(sd, "OutputStream", _MockOutputStream, raising=False)

    src = MicSource(sample_rate=16000, chunk_ms=10)
    sink = SpeakerSink(sample_rate=16000)

    async def _run_once():  # noqa: D401
        async for frame in src.frames():
            await sink.consume(frame)
            break  # only one frame for test

    await asyncio.wait_for(_run_once(), timeout=1) 