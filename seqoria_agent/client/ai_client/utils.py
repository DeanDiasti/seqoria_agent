from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from typing import Any, Optional

# Phase 9 constant extraction
from seqoria_agent.client.ai_client.consts_runtime import XML_ERROR_TAG  # noqa: F401

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
        Hexadecimal hash string (16 chars for xxhash, 8 for SHA-1).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if _xxhash is not None:
        return _xxhash.xxh3_64_hexdigest(data)
    else:
        return hashlib.sha1(data).hexdigest()[:16]


__all__: list[str] = [
    "null_async_cm",
    "extract_error_from_xml",
    "json_dumps",
    "json_loads",
    "json_impl_name",
    "fast_hash",
]


@asynccontextmanager
async def null_async_cm():
    """A no-op async context-manager used as a drop-in when no semaphore is set."""
    yield


def extract_error_from_xml(xml_str: str) -> Optional[dict[str, Any]]:  # noqa: D401
    """Parse *xml_str* for a `<error>` element.

    Returns a structured dict with keys ``code``, ``retryable``, ``detail`` and
    optional ``meta``.  If the input contains no error element or is malformed
    XML, ``None`` is returned.
    """
    # Fast check to avoid the heavier XML parser when there's clearly no error tag
    if f"<{XML_ERROR_TAG}" not in xml_str:
        return None

    try:
        # Lazy import: XML parsing is relatively rare compared to JSON handling.
        import xml.etree.ElementTree as ET

        root = ET.fromstring(text=xml_str.strip())
        error_el = root.find(path=XML_ERROR_TAG)
        if error_el is None:
            return None

        code = error_el.attrib.get("code", "UNKNOWN")
        retryable = error_el.attrib.get("retryable", "false").lower() == "true"
        detail_text = (error_el.text or "").strip()

        meta_node = error_el.find(path="meta")
        meta: Any | None = None
        if meta_node is not None and meta_node.text:
            try:
                meta = json_loads(meta_node.text)
            except Exception:
                meta = meta_node.text.strip()

        return {"code": code, "retryable": retryable, "detail": detail_text, "meta": meta}
    except Exception:
        return None
