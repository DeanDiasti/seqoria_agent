from __future__ import annotations

"""MCPx – eXtended Multi-Channel Protocol (v2).

Public API surface. Import *only* from here in downstream projects:

    from mcpx import AIClient, XMCP, DisplayEnvelope
"""

from mcpx.client import AIClient, ChatChunk  # noqa: F401
from mcpx.server.runtime import XMCP  # noqa: F401
from mcpx.models.envelope import DisplayEnvelope  # noqa: F401

__all__: list[str] = [
    "AIClient",
    "ChatChunk",
    "XMCP",
    "DisplayEnvelope",
] 