"""Core *seqoria_agent* client API.

For the initial refactor we *wrap* the proven implementation from the legacy
``seqoria_agent`` package so that external callers can immediately switch their import
paths while we incrementally extract the logic into smaller modules.

This keeps backwards-compatibility and avoids a huge, risky copy-paste move in
one go.  Future commits will replace this shim with a native implementation
that lives fully inside the ``seqoria_agent.client`` namespace.
"""

from __future__ import annotations

from seqoria_agent.client.client import AIClient
from seqoria_agent.models.chunk import ChatChunk  # noqa: F401

# Public re-exports -----------------------------------------------------------

__all__: list[str] = [
    "AIClient",
    "ChatChunk",
] 