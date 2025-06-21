"""Core *qutie* client API.

For the initial refactor we *wrap* the proven implementation from the legacy
``qutie`` package so that external callers can immediately switch their import
paths while we incrementally extract the logic into smaller modules.

This keeps backwards-compatibility and avoids a huge, risky copy-paste move in
one go.  Future commits will replace this shim with a native implementation
that lives fully inside the ``qutie.client`` namespace.
"""

from __future__ import annotations

from qutie.client.client import AIClient
from qutie.models.chunk import ChatChunk  # noqa: F401

# Public re-exports -----------------------------------------------------------

__all__: list[str] = [
    "AIClient",
    "ChatChunk",
] 