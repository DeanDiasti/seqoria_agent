from __future__ import annotations

import hashlib
from typing import Any

# ---------------------------------------------------------------------------
# High-performance JSON utilities with fallback
# ---------------------------------------------------------------------------
# Priority: orjson (fastest) -> ujson (fast) -> json (stdlib)

_json_impl: str = "json"  # Track which implementation is active

try:
    import orjson as _orjson  # type: ignore

    _json_impl = "orjson"
except ModuleNotFoundError:
    _orjson = None

try:
    import ujson as _ujson  # type: ignore

    if _orjson is None:
        _json_impl = "ujson"
except ModuleNotFoundError:
    _ujson = None

# Always import stdlib json as fallback
import json as _json_stdlib

# Try to import xxhash for fast hashing
try:
    import xxhash as _xxhash
except ModuleNotFoundError:
    _xxhash = None


def json_dumps(obj: Any, *, sort_keys: bool = False, indent: int | None = None) -> str:
    """Serialize object to JSON string using fastest available library.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    sort_keys : bool
        Whether to sort dictionary keys (for deterministic output).
    indent : int | None
        Indentation level for pretty printing. None for compact output.

    Returns
    -------
    str
        JSON string representation.
    """
    if _orjson is not None:
        opts = _orjson.OPT_SORT_KEYS if sort_keys else 0
        if indent:
            opts |= _orjson.OPT_INDENT_2
        return _orjson.dumps(obj, option=opts).decode("utf-8")
    elif _ujson is not None:
        return _ujson.dumps(obj, sort_keys=sort_keys, indent=indent)
    else:
        separators = (",", ":") if indent is None else None
        return _json_stdlib.dumps(obj, sort_keys=sort_keys, indent=indent, separators=separators)


def json_loads(s: str | bytes) -> Any:
    """Deserialize JSON string using fastest available library.

    Parameters
    ----------
    s : str | bytes
        JSON string or bytes to deserialize.

    Returns
    -------
    Any
        Deserialized Python object.
    """
    if _orjson is not None:
        return _orjson.loads(s)
    elif _ujson is not None:
        return _ujson.loads(s)
    else:
        return _json_stdlib.loads(s)


def json_impl_name() -> str:
    """Return the name of the active JSON implementation."""
    return _json_impl


def fast_hash(data: str | bytes) -> str:
    """Compute fast non-cryptographic hash of data.

    Uses xxhash if available, falls back to SHA-1 truncated.

    Parameters
    ----------
    data : str | bytes
        Data to hash.

    Returns
    -------
    str
        Hexadecimal hash string (16 characters).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if _xxhash is not None:
        return _xxhash.xxh3_64_hexdigest(data)
    else:
        return hashlib.sha1(data).hexdigest()[:16]


__all__: list[str] = [
    "json_dumps",
    "json_loads",
    "json_impl_name",
    "fast_hash",
]
