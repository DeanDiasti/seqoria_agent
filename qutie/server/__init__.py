"""Server-side helpers and decorators for building qutie-powered FastMCP servers."""

from .fastqutie import FastQutie as QUTIE  # noqa: F401

# Backwards compatibility export – FastQutie is the primary surface.
FastQutie = QUTIE  # type: ignore

__all__: list[str] = [
    "QUTIE",
    "FastQutie",
] 