from __future__ import annotations

"""Formatter registry & cache shared by XMCP server helpers.

This lightweight module centralises XML *formatter* selection so the various
server façades (FastXMCP, FastXMCPRouter, LocalToolbox) no longer duplicate the
same logic.  It is **framework-agnostic** and depends only on the generic
formatter utilities already shipped with *seqoria_agent*.

The public helpers are:
• register_formatter(py_type, formatter) – plug-in custom mapping
• get_formatter(sample, strict_mode=False)     – retrieve cached instance

Both functions are intentionally tiny (< 80 LOC total) and avoid importing
FastAPI, Starlette or Pydantic to keep the core portable.
"""

from typing import Any, Type

from seqoria_agent.server.formatters import (
    BaseXMLFormatter,
    DeclarativeXMLFormatter,
    derive_formatter_from_data,
)

__all__: list[str] = [
    "register_formatter",
    "get_formatter",
]

# ---------------------------------------------------------------------------
# Internal state – kept private to this module
# ---------------------------------------------------------------------------

# Cache key: (type(sample), strict_mode) → formatter instance
_CACHE: dict[tuple[Type[Any], bool], BaseXMLFormatter] = {}
# Explicit user-provided overrides: PyType → formatter instance
_CUSTOM: dict[Type[Any], BaseXMLFormatter] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_formatter(py_type: Type[Any], formatter: BaseXMLFormatter) -> None:  # noqa: D401
    """Register *formatter* for values whose concrete type is *py_type*.

    The mapping takes precedence over the built-in heuristic derivation.  A
    *single* instance of *formatter* is reused for all future calls and is NOT
    copied or re-instantiated.  Callers must therefore treat the object as
    **stateless**.
    """

    _CUSTOM[py_type] = formatter
    # Flush cache so new registration is respected on subsequent look-ups.
    _CACHE.clear()


def get_formatter(sample: Any, *, strict_mode: bool = False) -> BaseXMLFormatter:  # noqa: D401, ANN001
    """Return a formatter instance suitable for *sample*.

    The decision order is:
    1. explicit mapping registered via :pyfunc:`register_formatter`
    2. heuristic :pyfunc:`derive_formatter_from_data`
    3. generic :class:`DeclarativeXMLFormatter` fallback

    Instances are cached by ``(type(sample), strict_mode)`` so we create at
    most one per combination.  This keeps memory footprint minimal while
    avoiding repeated type inspection costs.
    """

    key = (type(sample), bool(strict_mode))
    if key in _CACHE:
        return _CACHE[key]

    # 1) explicit override registered by the host application
    if type(sample) in _CUSTOM:  # noqa: E721 – intentional type() comparison
        fmt = _CUSTOM[type(sample)]  # type: ignore[index]
    # 2) attempt to auto-derive from sample structure (dict heuristics)
    else:
        fmt = derive_formatter_from_data(sample=sample) or DeclarativeXMLFormatter()

    # Cache only when the formatter is *generic* for the data type.  For
    # plain ``dict`` values the derived formatter depends on the concrete
    # *keys* of the sample – caching merely by ``dict`` would re-use the
    # first structure for all subsequent calls and drop fields (see
    # tests/test_per_router_auth.py failure).  Therefore we skip caching
    # when the sample is a plain dict.  Custom user-registered instances
    # *are* cached because they come from the explicit mapping above.

    if not isinstance(sample, dict):
        _CACHE[key] = fmt

    return fmt 