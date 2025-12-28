"""Tests for speech-related data models."""

from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk, SpeechEvent


def test_transcript_chunk_creation_and_equality():
    chunk1 = TranscriptChunk(
        text="hello world",
        is_final=False,
        start_ts=0.0,
        end_ts=0.5,
        role="user",
    )
    chunk2 = TranscriptChunk(
        text="hello world",
        is_final=False,
        start_ts=0.0,
        end_ts=0.5,
        role="user",
    )

    # Basic attribute checks
    assert chunk1.text == "hello world"
    assert not chunk1.is_final
    assert chunk1.role == "user"

    # Equality & repr sanity
    assert chunk1 == chunk2
    assert "TranscriptChunk" in repr(chunk1)


def test_audio_chunk_creation_and_equality():
    pcm_data = b"\x00\x01\x02\x03"
    chunk1 = AudioChunk(
        pcm=pcm_data,
        sample_rate=16000,
        channels=1,
    )
    chunk2 = AudioChunk(
        pcm=pcm_data,
        sample_rate=16000,
        channels=1,
    )

    # Basic attribute checks
    assert chunk1.sample_rate == 16000
    assert chunk1.channels == 1

    # Equality & repr sanity
    assert chunk1 == chunk2
    assert "AudioChunk" in repr(chunk1)


def test_speech_event_literal():
    # Simple type check – ensure values are acceptable literals
    events: list[SpeechEvent] = ["user_start", "user_stop", "assistant_start", "assistant_stop"]
    assert set(events) == {"user_start", "user_stop", "assistant_start", "assistant_stop"} 