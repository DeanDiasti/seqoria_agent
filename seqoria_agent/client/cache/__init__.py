"""Caching utilities for *seqoria_agent* client package.

This file is largely copied from the original `seqoria_agent.cache` module but split
under the ``seqoria_agent.client.cache`` namespace so that different back-ends can live
side-by-side.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional, Protocol

__all__: list[str] = [
    "CacheProtocol",
    "NullCache",
    "DictCache",
    "RedisCache",
    "build_cache_key",
]


class CacheProtocol(Protocol):
    """Minimal async cache contract accepted by :class:`seqoria_agent.client.core.AIClient`."""

    async def get(self, key: str) -> Any | None:  # noqa: D401 – single-line docstring
        """Return the cached value or ``None`` when missing/expired."""

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: D401
        """Store *value* under *key* with optional *ttl* (seconds)."""

    async def delete(self, key: str) -> None:  # noqa: D401
        """Remove *key* from cache if present (best-effort)."""


# ---------------------------------------------------------------------------
# Fallback / test implementations
# ---------------------------------------------------------------------------


class NullCache:
    """No-op implementation that fulfils :class:`CacheProtocol`."""

    async def get(self, key: str):  # type: ignore[override]
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None):  # noqa: D401, ANN001
        return None

    async def delete(self, key: str):  # noqa: D401
        return None


class DictCache:
    """Simple in-memory cache with optional TTL support.

    Thread-safe across coroutines via an ``asyncio.Lock``.  **Not** suitable for
    multi-process deployments – use :class:`RedisCache` instead.
    """

    def __init__(self):
        self._data: Dict[str, tuple[Any, float | None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str):  # noqa: D401
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires_at = item
            if expires_at is not None and expires_at < time.time():
                # Expired – remove lazily
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None):  # noqa: D401, ANN001
        async with self._lock:
            expires_at: float | None = None
            if ttl is not None and ttl > 0:
                expires_at = time.time() + ttl
            self._data[key] = (value, expires_at)

    async def delete(self, key: str):  # noqa: D401
        async with self._lock:
            self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Redis backend (optional dependency)
# ---------------------------------------------------------------------------


class RedisCache:  # pragma: no cover – optional path
    """Redis-based cache using ``redis.asyncio``.

    Requires ``pip install redis>=4.5``.
    """

    _redis_missing_msg = (
        "RedisCache requires the 'redis' package. Install via 'pip install redis' to use this backend."
    )

    def __init__(self, url: str = "redis://localhost:6379", ttl: int | None = None):
        """Initialize the RedisCache.

        Parameters
        ----------
        url : str, optional
            The connection URL for the Redis server,
            by default "redis://localhost:6379".
        ttl : int | None, optional
            The default time-to-live in seconds for cache entries. If None,
            keys do not expire by default. Defaults to `None`.

        Raises
        ------
        ImportError
            If the `redis` package is not installed.
        """
        try:
            from redis import asyncio as redis_asyncio  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError(self._redis_missing_msg) from exc

        self._redis = redis_asyncio.from_url(url, encoding="utf-8", decode_responses=False)
        self._default_ttl = ttl

    async def get(self, key: str):  # noqa: D401
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None):  # noqa: D401, ANN001
        try:
            raw = json.dumps(value)
        except TypeError:
            raw = str(value)

        ttl_to_use = ttl if ttl is not None else self._default_ttl
        if ttl_to_use:
            await self._redis.setex(key, ttl_to_use, raw)
        else:
            await self._redis.set(key, raw)

    async def delete(self, key: str):  # noqa: D401
        await self._redis.delete(key)


# ---------------------------------------------------------------------------
# Helper – canonical cache keys
# ---------------------------------------------------------------------------

def build_cache_key(tool_name: str, arguments: Dict[str, Any], *, tenant_id: str | None = None) -> str:  # noqa: D401
    """Return a deterministic cache key for *tool_name* + *arguments*.

    Parameters
    ----------
    tool_name:
        Name of the tool.
    arguments:
        Input arguments dict (order independent).
    tenant_id:
        Optional namespace prefix for multi-tenant deployments. When given,
        the resulting key is ``tenant:{tenant_id}:tool:{tool_name}:{hash}``.
    """

    canonical_json = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    args_hash = hashlib.sha256(canonical_json.encode()).hexdigest()

    prefix = f"tenant:{tenant_id}:" if tenant_id else ""
    return f"{prefix}tool:{tool_name}:{args_hash}" 