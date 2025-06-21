from __future__ import annotations

"""fastxmcp package - public re-exports and convenience imports.

This sub-package provides the `FastXMCP` runtime::

    from seqoria_agent.server.fastxmcp import FastXMCP
"""

from .app import FastXMCP  # noqa: F401


__all__: list[str] = [
    "FastXMCP",
] 