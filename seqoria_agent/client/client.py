"""DEPRECATED shim for backwards compatibility.

Use `from seqoria_agent.client import AIClient` instead.
This module re-exports the new orchestrator while emitting a warning.
"""

from __future__ import annotations

import warnings

from seqoria_agent.client.ai_client.base import AIClient as _AIClient

warnings.warn(
    "seqoria_agent.client.client.AIClient is deprecated; import from seqoria_agent.client instead",
    DeprecationWarning,
    stacklevel=2,
)

AIClient = _AIClient

__all__ = ["AIClient"]