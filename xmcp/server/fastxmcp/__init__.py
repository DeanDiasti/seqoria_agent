from __future__ import annotations

"""fastxmcp package – public re-exports and convenience imports.

This sub-package replaces the historical ``fastxmcp.py`` module while
keeping the external import path **unchanged**::

    from xmcp.server.fastxmcp import FastXMCP

Down-stream projects therefore do **not** need to update their code.
"""

from .app import FastXMCP  # noqa: F401

__all__: list[str] = [
    "FastXMCP",
] 