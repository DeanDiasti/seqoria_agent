from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from typing import Any, Optional, Dict, Any
import hashlib
import xxhash

# Phase 9 constant extraction
from seqoria_agent.client.ai_client.consts_runtime import XML_ERROR_TAG  # noqa: F401

# Phase-2 optimisation: use orjson for faster canonical dumps when available
try:
    import orjson as _json_fast  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    _json_fast = None

__all__: list[str] = ["null_async_cm", "extract_error_from_xml", "build_cache_key"]


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
                meta = json.loads(s=meta_node.text)
            except Exception:
                meta = meta_node.text.strip()

        return {"code": code,
                "retryable": retryable,
                "detail": detail_text,
                "meta": meta}
    except Exception:
        return None


def build_cache_key(tool_name: str, arguments: Dict[str, Any], *, tenant_id: str | None = None) -> str:  # noqa: D401
    """Return a deterministic cache key for *tool_name* + *arguments*.
    
    Parameters
    ----------
    tool_name:
        Name of the tool.
    arguments:
        Input arguments dict (order independent).
    tenant_id:
        Optional namespace prefix for multi-tenant deployments. When given, the
        resulting key is ``tenant:{tenant_id}:tool:{tool_name}:{hash}``.
    """
    if _json_fast is not None:
        # OPT_SORT_KEYS yields deterministic ordering equivalent to sort_keys=True
        canonical_json = _json_fast.dumps(arguments, option=_json_fast.OPT_SORT_KEYS).decode()
    else:
        canonical_json = json.dumps(obj=arguments, sort_keys=True, separators=(",", ":"))

    # Try faster non-cryptographic hash when available ----------------------
    try:
        args_hash = xxhash.xxh3_64_hexdigest(canonical_json)
    except Exception:
        # Fallback to built-in SHA-256 to avoid hard dependency
        args_hash = hashlib.sha256(string=canonical_json.encode()).hexdigest()

    prefix = f"tenant:{tenant_id}:" if tenant_id else ""
    return f"{prefix}tool:{tool_name}:{args_hash}" 