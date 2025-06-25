from __future__ import annotations

"""fastxmcp package – public re-exports.

Canonical usage::

    from seqoria_agent.server.fastxmcp import FastXMCP, FastXMCPRouter
"""

from seqoria_agent.server.fastxmcp_public import FastXMCP, FastXMCPRouter  # noqa: F401

__all__: list[str] = [
    "FastXMCP",
    "FastXMCPRouter",
] 