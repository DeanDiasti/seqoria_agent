"""Regressions for message round trips and cache value isolation."""

from types import SimpleNamespace

import pytest

from seqoria_agent.client.cache import DictCache, RedisCache
from seqoria_agent.models import (
    AssistantMessage, LLMUsage, SystemMessage, ToolCallError, ToolCallRequest,
    ToolCallRequestList, ToolMessages, UserMessage,
)


def test_message_defaults_round_trip():
    for model in (AssistantMessage, SystemMessage, UserMessage, LLMUsage, ToolCallRequestList):
        message = model()
        assert model.model_validate(message.model_dump()) == message
    assert UserMessage().role == "user"
    assert SystemMessage().role == "system"
    assert AssistantMessage().role == "assistant"
    assert LLMUsage().total_tokens == 0


def test_tool_messages_preserve_generators_and_errors():
    request = ToolCallRequest(id="call", name="example", arguments={})
    error = ToolCallError(tool_call_id="call", name="example", error="failed")
    messages = ToolMessages(item for item in [request])
    messages.extend(item for item in [error])
    assert messages == [request, error]
    messages += (item for item in [request])
    assert messages == [request, error, request]


def test_tool_messages_validate_updates_atomically():
    error = ToolCallError(tool_call_id="call", name="example", error="failed")
    messages = ToolMessages([error])
    for update in (
        lambda: messages.extend(iter([error, "invalid"])),
        lambda: messages.insert(0, "invalid"),
        lambda: messages.__setitem__(0, "invalid"),
        lambda: messages.__setitem__(slice(None), iter([error, "invalid"])),
    ):
        with pytest.raises(TypeError):
            update()
        assert messages == [error]


@pytest.mark.asyncio
async def test_dict_cache_copies_nested_values():
    cache = DictCache()
    original = {"items": [{"value": 1}]}
    await cache.set("key", original)
    original["items"][0]["value"] = 2
    first = await cache.get("key")
    assert first == {"items": [{"value": 1}]}
    first["items"].clear()
    assert await cache.get("key") == {"items": [{"value": 1}]}


@pytest.mark.asyncio
async def test_cache_expiration_boundary(monkeypatch):
    now = 100.0
    monkeypatch.setattr("seqoria_agent.client.cache.time.monotonic", lambda: now)
    cache = DictCache()
    await cache.set("key", "value", ttl=1)
    now = 101.0
    assert await cache.get("key") is None
    with pytest.raises(ValueError, match="ttl"):
        await cache.set("key", "value", ttl=-1)


@pytest.mark.asyncio
async def test_redis_cache_preserves_json_types_and_rejects_lossy_values(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.data = {}

        async def get(self, key):
            return self.data.get(key)

        async def set(self, key, value):
            self.data[key] = value.encode()

    backend = FakeRedis()
    monkeypatch.setattr(
        "seqoria_agent.client.cache.redis_asyncio",
        SimpleNamespace(from_url=lambda *args, **kwargs: backend),
    )
    cache = RedisCache()
    await cache.set("key", {"items": [1, True, None]})
    assert await cache.get("key") == {"items": [1, True, None]}
    with pytest.raises(TypeError):
        await cache.set("key", object())
    assert await cache.get("key") == {"items": [1, True, None]}
    with pytest.raises(ValueError, match="ttl"):
        await cache.set("key", "value", ttl=-1)
    with pytest.raises(ValueError, match="ttl"):
        RedisCache(ttl=-1)


def test_settings_copy_detaches_configuration_containers():
    from seqoria_agent.client.settings import ClientSettings
    from seqoria_agent import LocalToolbox

    toolbox = LocalToolbox()
    settings = ClientSettings(
        urls=["https://example.invalid/mcp?secret=credential"],
        local_toolbox=[toolbox],
        retry_policy={"max_attempts": 2},
    )
    copied = settings.copy()
    copied.urls.clear()
    copied.local_toolbox.clear()
    copied.retry_policy["max_attempts"] = 3
    assert settings.urls == ["https://example.invalid/mcp?secret=credential"]
    assert settings.local_toolbox == [toolbox]
    assert settings.retry_policy == {"max_attempts": 2}
    assert "credential" not in repr(settings)


@pytest.mark.parametrize("max_entries", [0, -1, 1.5, float("nan"), float("inf"), True])
def test_dict_cache_rejects_invalid_capacity(max_entries):
    with pytest.raises(ValueError, match="positive integer"):
        DictCache(max_entries=max_entries)
