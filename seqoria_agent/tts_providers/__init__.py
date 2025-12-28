"""Text-to-speech provider interfaces (Phase 1 stub)."""
from seqoria_agent.tts_providers.base import BaseTTSAdapter  # noqa: F401
from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS  # noqa: F401

__all__: list[str] = ["BaseTTSAdapter",
                      "ElevenLabsTTS"] 