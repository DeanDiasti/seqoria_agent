from __future__ import annotations

"""Inline tool support (LocalToolbox).

A *LocalToolbox* lets developers define `@tool` / `@display_tool` functions in
regular Python modules without running an HTTP server.  An :class:`AIClient`
instance can discover these tools and invoke them in-process exactly like it
would call remote FastXMCP tools.
"""

from .local import LocalToolbox  # noqa: F401

__all__: list[str] = [
    "LocalToolbox",
] 