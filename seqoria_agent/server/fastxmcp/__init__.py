from __future__ import annotations

"""Public FastXMCP interfaces (canonical import paths).

Example
-------
>>> from seqoria_agent.server.fastxmcp_public import FastXMCP, FastXMCPRouter
"""

from seqoria_agent.server.fastxmcp.app import FastXMCP  # noqa: F401
from seqoria_agent.server.fastxmcp.router import FastXMCPRouter  # noqa: F401

__all__: list[str] = ["FastXMCP",
                      "FastXMCPRouter"] 