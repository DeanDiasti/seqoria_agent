"""Tests for the ElevenLabsTTS adapter (Phase 3).

These tests stub the external *elevenlabs* SDK so they run without the
optional extra being installed and without making any network calls.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import AsyncIterator, List

import pytest

from seqoria_agent.models.speech_models import AudioChunk

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_elevenlabs(monkeypatch):  # noqa: D401
    """Provide a dummy *elevenlabs* module with minimal SDK surface.

    The adapter imports ``elevenlabs.ElevenLabs`` and expects an instance
    exposing ``text_to_speech.convert_realtime`` that yields *PCM* bytes.  A
    lightweight stub is injected into *sys.modules* **before** the adapter is
    (re-)imported so that the normal import path succeeds even when the real
    package is not installed.
    """

    mod = types.ModuleType("elevenlabs")

    class _DummyTTS:  # noqa: D401 – minimal stub for the SDK helper
        def convert_realtime(self, *, voice_id=None, text=None, **_):  # noqa: ANN001, D401
            # Emit one fake PCM buffer for each text fragment encountered.
            for _ in text or []:
                yield b"\x00\x01"

        # Used by ElevenLabsTTS.generate()
        def convert(self, *, voice_id=None, text=None, **_):  # noqa: ANN001, D401
            # Return an iterator (not generator) that yields a fixed PCM buffer
            # to simulate the SDK's blocking convert() helper.
            return iter([b"\x00\x02", b"\x00\x03"])

    class _DummyElevenLabs:  # noqa: D401 – mimics the SDK client root
        def __init__(self, api_key=None):  # noqa: D401, ANN001
            self.text_to_speech = _DummyTTS()

    mod.ElevenLabs = _DummyElevenLabs  # type: ignore[attr-defined]

    # Insert (or replace) the stub in the import registry.
    monkeypatch.setitem(sys.modules, "elevenlabs", mod)

    # Ensure the adapter picks up the stubbed module – reload if already imported.
    if "seqoria_agent.tts_providers.elevenlabs_adapter" in sys.modules:
        importlib.reload(sys.modules["seqoria_agent.tts_providers.elevenlabs_adapter"])
    else:
        importlib.import_module("seqoria_agent.tts_providers.elevenlabs_adapter")

    return mod


async def _text_iter() -> AsyncIterator[str]:  # noqa: D401
    """Single-item async iterator used by the stream test."""
    yield "Hello world"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_audio(stub_elevenlabs, monkeypatch):  # noqa: D401
    """Adapter should emit *AudioChunk* objects for each PCM buffer."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    # Prevent background HTTP fetch in _ensure_voice_cache
    async def _noop(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    monkeypatch.setattr(ElevenLabsTTS, "_ensure_voice_cache", _noop, raising=True)

    tts = ElevenLabsTTS(api_key="dummy")

    chunks: List[AudioChunk] = []
    async for ch in tts.stream(_text_iter()):
        chunks.append(ch)

    assert chunks, "Adapter did not yield any AudioChunk instances"
    assert all(isinstance(c, AudioChunk) for c in chunks)
    assert all(c.sample_rate == 24000 for c in chunks)
    assert all(c.channels == 1 for c in chunks)


def test_supports_voice_cache(stub_elevenlabs, monkeypatch):  # noqa: D401
    """supports_voice should honour the internal voice cache when populated."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    # Empty cache → permissive
    monkeypatch.setattr(ElevenLabsTTS, "_voice_cache", set(), raising=True)
    assert ElevenLabsTTS.supports_voice("foo") is True

    # Populated cache → strict membership test
    monkeypatch.setattr(ElevenLabsTTS, "_voice_cache", {"alice", "bob"}, raising=True)
    assert ElevenLabsTTS.supports_voice("alice") is True
    assert ElevenLabsTTS.supports_voice("charlie") is False


def test_api_key_required(stub_elevenlabs, monkeypatch):  # noqa: D401
    """Instantiation without an API key should raise *ValueError*."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    # Ensure env var is absent so constructor must rely on explicit argument.
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    async def _noop(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    monkeypatch.setattr(ElevenLabsTTS, "_ensure_voice_cache", _noop, raising=True)

    with pytest.raises(ValueError):
        ElevenLabsTTS(api_key=None) 


# ---------------------------------------------------------------------------
# New tests for the *generate* helper (one-shot synthesis)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_pcm(stub_elevenlabs, monkeypatch):  # noqa: D401
    """generate() should return raw PCM bytes for the given text."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    async def _noop(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    # Skip background voice cache refresh
    monkeypatch.setattr(ElevenLabsTTS, "_ensure_voice_cache", _noop, raising=True)

    tts = ElevenLabsTTS(api_key="dummy")

    pcm = await tts.generate("Hello world")

    # The stub returns two buffers joined together by the adapter.
    assert pcm == b"\x00\x02\x00\x03"


@pytest.mark.asyncio
async def test_generate_empty_text_returns_empty(stub_elevenlabs, monkeypatch):  # noqa: D401
    """generate() should return *empty* bytes when text is empty."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    async def _noop(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    monkeypatch.setattr(ElevenLabsTTS, "_ensure_voice_cache", _noop, raising=True)

    tts = ElevenLabsTTS(api_key="dummy")

    pcm = await tts.generate("")
    assert pcm == b"" 


# ---------------------------------------------------------------------------
# Additional tests for new parameters (voice settings & output_format)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_output_format_sample_rate(stub_elevenlabs, monkeypatch):  # noqa: D401
    """stream() should derive sample_rate from *output_format* (e.g. pcm_16000)."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    async def _noop(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    monkeypatch.setattr(ElevenLabsTTS, "_ensure_voice_cache", _noop, raising=True)

    tts = ElevenLabsTTS(api_key="dummy", output_format="pcm_16000")

    chunks: List[AudioChunk] = []
    async for ch in tts.stream(_text_iter()):
        chunks.append(ch)

    assert chunks, "No AudioChunk emitted with custom output_format"
    assert all(c.sample_rate == 16000 for c in chunks), "sample_rate not derived from output_format"


@pytest.mark.asyncio
async def test_non_pcm_output_format_raises(stub_elevenlabs, monkeypatch):  # noqa: D401
    """stream() should raise when *output_format* is not PCM based."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    async def _noop(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    monkeypatch.setattr(ElevenLabsTTS, "_ensure_voice_cache", _noop, raising=True)

    tts = ElevenLabsTTS(api_key="dummy", output_format="mp3_44100_128")

    with pytest.raises(RuntimeError):
        async for _ in tts.stream(_text_iter()):  # pragma: no cover – expected error
            pass


@pytest.mark.asyncio
async def test_voice_settings_preserved(stub_elevenlabs, monkeypatch):  # noqa: D401
    """Constructor should retain provided voice setting overrides."""

    from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS

    async def _noop(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    monkeypatch.setattr(ElevenLabsTTS, "_ensure_voice_cache", _noop, raising=True)

    tts = ElevenLabsTTS(
        api_key="dummy",
        stability=0.5,
        similarity_boost=0.9,
        style=0.3,
        speed=1.1,
        use_speaker_boost=True,
    )

    expected = {
        "stability": 0.5,
        "similarity_boost": 0.9,
        "style": 0.3,
        "speed": 1.1,
        "use_speaker_boost": True,
    }

    assert tts._voice_settings == expected 