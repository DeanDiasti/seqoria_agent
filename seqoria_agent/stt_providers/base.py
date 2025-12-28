"""Base interface for speech-to-text adapters (Phase 2).

Adapters convert incoming ``AudioChunk`` streams into incremental
``TranscriptChunk`` objects.  Concrete implementations may call remote APIs
(e.g. Whisper, Deepgram) or run local models, but they all share the same
async generator contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk


class BaseSTTAdapter(ABC):
    """Stream speech-to-text partials until the source exhausts."""

    @abstractmethod
    async def stream(self,
                     frames: AsyncIterator[AudioChunk],
                     *,
                     language: str | None = None) -> AsyncIterator[TranscriptChunk]:
        """Yield :class:`TranscriptChunk` objects for each recognised segment."""
        raise NotImplementedError("This STT adapter does not support streaming transcription.")

    # ---------------------------------------------------------------------
    # Capability helpers ---------------------------------------------------
    # ---------------------------------------------------------------------

    @classmethod
    def supports_language(cls, lang: str) -> bool:  # noqa: D401
        """Return *True* when *lang* is supported by the adapter.

        The default implementation returns ``True`` for all languages -
        concrete subclasses should override when they have limited language
        coverage.
        """
        return True

    # ---------------------------------------------------------------------
    # New one-shot transcription API --------------------------------------
    # ---------------------------------------------------------------------

    async def transcribe(self,
                         pcm: bytes,
                         sample_rate: int,
                         channels: int,
                         *,
                         language: str | None = None) -> list["TranscriptChunk"]:
        """One-shot transcription helper.

        Concrete adapters can optionally override this method to provide a
        blocking/non-streaming transcription path.  The *base* implementation
        raises ``NotImplementedError`` so that subclasses are *not required*
        to implement it for interface compliance - this matches the behaviour
        expected by the unit-tests which instantiate dummy adapters lacking a
        ``transcribe`` implementation.
        """
        raise NotImplementedError("This STT adapter does not support one-shot 'transcribe' calls.") 