"""Server-side helpers and decorators for building xmcp-powered FastMCP servers."""

from .fastxmcp import FastXMCP as XMCP  # noqa: F401

# Backwards compatibility export – FastXMCP is the primary surface.
FastXMCP = XMCP  # type: ignore

__all__: list[str] = [
    "XMCP",
    "FastXMCP",
] 