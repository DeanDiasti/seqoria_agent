"""Speech-to-text provider interfaces (Phase 1 stub)."""
from seqoria_agent.stt_providers.base import BaseSTTAdapter  # noqa: F401
from seqoria_agent.stt_providers.fireworks_stt import FireworksSTT  # noqa: F401

__all__: list[str] = ["BaseSTTAdapter",
                      "FireworksSTT"] 