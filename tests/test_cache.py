"""Unit tests for *xmcp.client.cache* helpers."""

import asyncio
import time

import pytest

from seqoria_agent.client.cache import DictCache, NullCache, build_cache_key


@pytest.mark.asyncio
async def test_dict_cache_set_get_delete():
    cache = DictCache()

    await cache.set("foo", "bar", ttl=1)
    assert await cache.get("foo") == "bar"

    # Wait for expiry
    await asyncio.sleep(1.1)
    assert await cache.get("foo") is None

    await cache.set("key", 123)
    assert await cache.get("key") == 123
    await cache.delete("key")
    assert await cache.get("key") is None


@pytest.mark.asyncio
async def test_null_cache_is_noop():
    cache = NullCache()
    await cache.set("x", "y")  # Should do nothing
    assert await cache.get("x") is None
    await cache.delete("x")  # Should not raise


def test_build_cache_key_deterministic():
    args1 = {"b": 2, "a": 1}
    args2 = {"a": 1, "b": 2}
    key1 = build_cache_key("tool", args1)
    key2 = build_cache_key("tool", args2)
    assert key1 == key2  # order shouldn't matter
    assert key1.startswith("tool:tool:") 