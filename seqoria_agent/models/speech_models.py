"""Core data classes and type aliases for Seqoria speech pipeline (Phase 1).

These models are intentionally lightweight and framework-agnostic so they can
be safely shared across client and server code as the speech pipeline evolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Public type aliases
SpeechEvent = Literal["user_start", "user_stop", "assistant_start", "assistant_stop"]

@dataclass(slots=True)
class TranscriptChunk:
    """Partial or final transcript of a speaker segment."""

    text: str
    is_final: bool
    start_ts: float  # seconds
    end_ts: float
    role: Literal["user", "assistant"] = "user"

    def __post_init__(self) -> None:  # noqa: D401
        if self.start_ts < 0 or self.end_ts < 0:
            raise ValueError("Timestamps must be non-negative.")
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must be >= start_ts.")


@dataclass(slots=True)
class AudioChunk:
    """PCM audio buffer accompanied by basic metadata."""

    pcm: bytes  # 16-bit LE or float32 PCM buffer
    sample_rate: int  # e.g. 16000
    channels: int = 1

    def __post_init__(self) -> None:  # noqa: D401
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        
