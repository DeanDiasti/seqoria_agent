from __future__ import annotations

"""xmcp – eXtended Multi-Channel Protocol (v2).

Public API surface. Import *only* from here in downstream projects:

    from xmcp import AIClient, XMCP, DisplayEnvelope
"""

from xmcp.client import AIClient, ChatChunk  # noqa: F401
from xmcp.server.runtime import XMCP  # noqa: F401
from xmcp.models.envelope import DisplayEnvelope  # noqa: F401

__all__: list[str] = [
    "AIClient",
    "ChatChunk",
    "XMCP",
    "DisplayEnvelope",
] 