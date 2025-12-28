"""Base interface for text-to-speech adapters (Phase 2).

Adapters convert chunks of text into PCM audio frames via an async generator
interface.  Implementations may call cloud TTS services or local engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from seqoria_agent.models.speech_models import AudioChunk


class BaseTTSAdapter(ABC):
    """Convert text fragments into audio chunks."""

    @abstractmethod
    async def stream(self,
                     text_iter: AsyncIterator[str],
                     *,
                     voice: str | None = None) -> AsyncIterator[AudioChunk]:
        """Yield :class:`AudioChunk` buffers representing the spoken text."""

    @abstractmethod
    async def generate(self,
                       text: str,
                       *,
                       voice: str | None = None) -> bytes:
        """Generate raw PCM bytes representing the spoken text."""
        

    # Capability helpers ------------------------------------------------
    @classmethod
    def supports_voice(cls, voice: str) -> bool:  # noqa: D401
        """Return *True* when *voice* is supported by the adapter.

        Concrete adapters should override this to advertise voice IDs they
        can generate.  The default implementation is permissive.
        """
        return True 