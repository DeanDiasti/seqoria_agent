"""Regression tests for shared web-app clients and the public package API."""

import asyncio
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

import seqoria_agent
from seqoria_agent import AIClient, AsyncAIClient, LocalToolbox, SeqoriaMCP
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.client.cache import DictCache
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.models.chunk import AssistantMessage, ToolCallRequest, ToolMessages
from seqoria_agent.models.errors import ToolRemoteError
from seqoria_agent.providers import AsyncBaseProvider, BaseProvider
from seqoria_agent.providers.errors import ConnectionError
from seqoria_agent.providers.schemas import LLMResponse, ToolCall, ToolCallList


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield await self.async_generate(**kwargs)


def request(name, call_id=None, **arguments):
    return ToolCallRequest(id=call_id or name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_shared_client_keeps_concurrent_histories_separate():
    async with AsyncAIClient(providers=[EchoProvider()]) as client:

        async def chat(text):
            history = []
            replies = [item async for item in client.chat(text, history)]
            return history, replies[-1].content

        (left, left_reply), (right, right_reply) = await asyncio.gather(chat("alice"), chat("bob"))
        assert left_reply == "alice"
        assert right_reply == "bob"
        assert left is not right
        assert len(left) == len(right) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_parallel_chat_results_stay_paired_with_their_requests(streaming):
    box = LocalToolbox()

    @box.tool()
    async def slow():
        await asyncio.sleep(0.02)
        return "slow-result"

    @box.tool()
    async def fast():
        return "fast-result"

    class ToolsProvider(EchoProvider):
        async def async_generate(self, *, chat_history, **kwargs):
            if not isinstance(chat_history[-1], AssistantMessage) and len(chat_history) == 2:
                return ToolCallList(
                    model_id="tools",
                    usage=None,
                    tool_calls=[
                        ToolCall(id="slow-id", function_name="slow", arguments={}),
                        ToolCall(id="fast-id", function_name="fast", arguments={}),
                    ],
                )
            return LLMResponse(model_id="tools", text="done", type="llm_response")

    async with AsyncAIClient(providers=[ToolsProvider()], local_toolbox=box) as client:
        history = []
        chat = client.stream_chat if streaming else client.chat
        events = [event async for event in chat("go", history)]
        tools = next(event for event in events if isinstance(event, ToolMessages))
        assert [(tools[i].id, tools[i + 1].tool_call_id) for i in (0, 2)] == [
            ("slow-id", "slow-id"),
            ("fast-id", "fast-id"),
        ]
        assert tools[1].result == "slow-result"
        assert tools[3].result == "fast-result"


@pytest.mark.asyncio
async def test_closing_tool_stream_cancels_remaining_async_tools():
    box = LocalToolbox()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @box.tool()
    async def slow():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    @box.tool()
    async def fast():
        await started.wait()
        return "ready"

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box) as client:
        stream = client.run_call_stream([request("slow"), request("fast")])
        req, _ = await anext(stream)
        assert req.name == "fast"
        await stream.aclose()
        assert cancelled.is_set()


@pytest.mark.asyncio
async def test_sync_tool_does_not_block_other_requests():
    import threading

    box = LocalToolbox()
    release = threading.Event()
    started = threading.Event()

    @box.tool()
    def blocking():
        started.set()
        return release.wait(timeout=2)

    task = asyncio.create_task(box.call_tool("blocking", {}))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert not task.done()
        # This coroutine can run while the synchronous tool is still waiting.
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_injection_does_not_rewrite_the_original_tool_request():
    box = LocalToolbox()

    @box.tool(server_populated=["user_id"])
    async def identity(user_id: str):
        return user_id

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box) as client:
        call = request("identity", user_id="forged")
        result = await client.run_tool_calls([call], context={"user_id": "trusted"})
        assert result[0].result == "trusted"
        assert call.arguments == {"user_id": "forged"}


@pytest.mark.asyncio
async def test_cache_isolates_users_permissions_and_client_instances():
    box = LocalToolbox()
    counter = 0

    @box.tool()
    async def value():
        nonlocal counter
        counter += 1
        return str(counter)

    cache = DictCache()
    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box, cache=cache) as client:

        async def invoke(user, permission="reader"):
            return (
                await client.run_tool_calls(
                    [request("value")],
                    context={
                        "tenant_id": "same",
                        "user_id": user,
                        "permission": permission,
                    },
                )
            )[0].result

        assert await invoke("alice") == "1"
        assert await invoke("bob") == "2"
        assert await invoke("alice") == "1"
        assert await invoke("alice", "admin") == "3"
    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box, cache=cache) as other:
        result = await other.run_tool_calls(
            [request("value")],
            context={
                "tenant_id": "same",
                "user_id": "alice",
                "permission": "reader",
            },
        )
        assert result[0].result == "4"


@pytest.mark.asyncio
async def test_dict_cache_is_bounded_and_evicts_least_recently_used():
    cache = DictCache(max_entries=2)
    await cache.set("a", 1)
    await cache.set("b", 2)
    assert await cache.get("a") == 1
    await cache.set("c", 3)
    assert await cache.get("b") is None
    assert await cache.get("a") == 1


def test_local_schema_supports_nested_and_optional_types():
    class Item(BaseModel):
        label: str

    box = LocalToolbox()

    @box.tool()
    def inspect_items(items: list[Item], mode: Literal["fast", "full"], count: int | None = None):
        return items

    schema = box.list_tools()[0].inputSchema
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["mode"]["enum"] == ["fast", "full"]
    assert schema["required"] == ["items", "mode"]
    assert "Item" in schema["$defs"]


@pytest.mark.asyncio
async def test_registry_refresh_keeps_old_routes_live_until_discovery_finishes():
    box = LocalToolbox()

    @box.tool()
    async def ping():
        return "pong"

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box) as client:
        started = asyncio.Event()
        finish = asyncio.Event()

        async def slow_list():
            started.set()
            await finish.wait()
            return SimpleNamespace(tools=[])

        session = SimpleNamespace(list_tools=slow_list)
        client.pool._connections.append(
            SimpleNamespace(session=session, url="fake", _stop_event=None, _owner_task=None)
        )
        refresh = asyncio.create_task(client.refresh_tools())
        await asyncio.wait_for(started.wait(), timeout=1)
        try:
            assert (await client.run_tool_calls([request("ping")]))[0].result == "pong"
        finally:
            finish.set()
            await refresh
            client.pool._connections.clear()


@pytest.mark.asyncio
async def test_schema_alias_routes_to_original_tool_name():
    first, second = LocalToolbox(), LocalToolbox()

    @first.tool(name="echo")
    async def one(value: str):
        return value

    @second.tool(name="echo")
    async def two(number: int):
        return number * 2

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=[first, second]) as client:
        alias = next(
            t["function"]["name"] for t in client.list_tools() if "__" in t["function"]["name"]
        )
        result = await client.run_tool_calls([request(alias, number=3)])
        assert result[0].result == "6"


@pytest.mark.asyncio
async def test_remote_tool_error_is_not_successfully_cached_or_retried():
    result = CallToolResult(isError=True, content=[TextContent(type="text", text="denied")])
    session = AsyncMock()
    session.call_tool.return_value = result
    pool = MCPConnectionPool(retry_policy={"max_attempts": 3})
    pool._tool_map["protected"] = session
    with pytest.raises(ToolRemoteError, match="denied"):
        await pool.call_tool("protected", {})
    assert session.call_tool.await_count == 1


@pytest.mark.asyncio
async def test_remote_server_resolves_hidden_parameters_with_real_mcp_context():
    server = SeqoriaMCP("injection")
    seen = []

    @server.register_param_provider
    def identity(ctx, tool, parameter):
        seen.append(ctx.request_context.request_id)
        return "authenticated-user"

    @server.tool(server_populated=["user_id"], depends_on=["authorize"])
    async def greet(name: str, user_id: str):
        return f"{name}:{user_id}"

    async with create_connected_server_and_client_session(server) as session:
        listed = await session.list_tools()
        tool = listed.tools[0]
        assert "user_id" not in tool.inputSchema["properties"]
        assert "ctx" not in tool.inputSchema["properties"]
        assert tool.annotations.model_dump()["depends_on"] == ["authorize"]
        result = await session.call_tool("greet", {"name": "alice", "user_id": "forged"})
        assert not result.isError
        assert result.content[0].text == "alice:authenticated-user"
        assert seen


@pytest.mark.asyncio
async def test_real_stdio_transport_can_close_from_a_different_request_task(tmp_path):
    script = tmp_path / "server.py"
    script.write_text("""from seqoria_agent import SeqoriaMCP
server = SeqoriaMCP("stdio")
@server.tool()
async def ping() -> str:
    return "pong"
server.run(transport="stdio")
""")
    config = MCPServerConfig(
        url="stdio://" + shlex.join([sys.executable, str(script)]),
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    client = AsyncAIClient(providers=[EchoProvider()], mcp_servers=[config])
    await asyncio.wait_for(asyncio.create_task(client.__aenter__()), timeout=10)
    try:
        assert (await client.run_tool_calls([request("ping")]))[0].result == "pong"
    finally:
        await asyncio.wait_for(asyncio.create_task(client.close()), timeout=10)
    assert not client.pool.connections
    assert client.pool._http_client_pool is None


@pytest.mark.asyncio
async def test_stream_fails_over_when_iteration_fails_before_output():
    class Broken(EchoProvider):
        async def async_stream(self, **kwargs):
            raise ConnectionError("broken")
            yield  # pragma: no cover

    llm = LLMInterface(providers=[Broken(), EchoProvider()])
    events = [event async for event in llm.stream([AssistantMessage(content="works")])]
    assert events[-1].content == "works"


@pytest.mark.asyncio
async def test_stream_does_not_switch_provider_after_output_is_visible():
    class Partial(EchoProvider):
        async def async_stream(self, **kwargs):
            yield LLMResponse(model_id="partial", text="first", type="llm_delta")
            raise ConnectionError("partial")

    llm = LLMInterface(providers=[Partial(), EchoProvider()])
    stream = llm.stream([AssistantMessage(content="must not appear")])
    assert (await anext(stream)).content == "first"
    with pytest.raises(ConnectionError):
        await anext(stream)


class BlockingProvider(BaseProvider):
    def generate(self, **kwargs):
        return LLMResponse(model_id="blocking", text="ok", type="llm_response")

    def stream(self, **kwargs):
        yield self.generate(**kwargs)


def test_sync_client_accepts_a_real_sync_provider_contract():
    with AIClient(providers=[BlockingProvider()]) as client:
        assert client.chat_sync("hello", history=[]).content == "ok"
        assert list(client.stream_chat("hello", history=[]))[-1].content == "ok"


@pytest.mark.asyncio
async def test_sync_facade_rejects_running_event_loop_cleanly():
    client = AIClient(providers=[BlockingProvider()])
    with pytest.raises(RuntimeError, match="Use AsyncAIClient"):
        client.close()


def test_automatic_memory_is_removed_from_public_api():
    assert "AgenticMemory" not in seqoria_agent.__all__
    assert not hasattr(seqoria_agent, "AgenticMemory")


@pytest.mark.asyncio
async def test_local_arguments_are_validated_and_nested_models_reach_the_tool():
    from pydantic import ValidationError

    class Payload(BaseModel):
        label: str

    box = LocalToolbox()

    @box.tool()
    async def typed(payload: Payload, count: int = 2):
        return payload.label * count

    assert (await box.call_tool("typed", {"payload": {"label": "a"}}))[0].text == "aa"
    with pytest.raises(ValidationError):
        await box.call_tool("typed", {"payload": {"label": "a"}, "count": "invalid"})
    with pytest.raises(ValidationError):
        await box.call_tool("typed", {"payload": {"label": "a"}, "undeclared": True})


@pytest.mark.asyncio
async def test_remote_missing_identity_fails_without_running_the_tool():
    server = SeqoriaMCP("missing-identity")
    called = False

    @server.tool(server_populated=["tenant_id"])
    async def protected(tenant_id: str = "unsafe-default"):
        nonlocal called
        called = True
        return tenant_id

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("protected", {"tenant_id": "forged"})
        assert result.isError
        assert not called


def test_settings_copy_and_explicit_overrides():
    from seqoria_agent.client.settings import ClientSettings

    settings = ClientSettings(providers=[EchoProvider()], refresh_interval=30, cache_ttl=42)
    copied = settings.with_provider(EchoProvider())
    assert len(settings.providers) == 1
    assert len(copied.providers) == 2
    client = AsyncAIClient(settings=settings, refresh_interval=0)
    assert client.registry._refresh_interval == 0
    assert client.registry._refresher_task is None
    assert client.executor._cache_ttl == 42


@pytest.mark.asyncio
async def test_generator_server_additions_and_removals_are_not_consumed_twice():
    async with AsyncAIClient(providers=[EchoProvider()]) as client:
        added = await client.add_mcp_servers(
            (url for url in ["https://example.invalid/mcp"]), connect=False
        )
        assert added == 1
        assert len(client.list_mcp_servers()) == 1
        removed = await client.remove_mcp_servers((url for url in ["https://example.invalid/mcp"]))
        assert removed == 1
        assert client.list_mcp_servers() == []


@pytest.mark.asyncio
async def test_provider_lifecycle_releases_owned_http_resources():
    class Provider(EchoProvider):
        def __init__(self):
            self.async_client = SimpleNamespace(aclose=AsyncMock())

    provider = Provider()
    async with provider:
        async with AsyncAIClient(providers=[provider]):
            pass
        provider.async_client.aclose.assert_not_awaited()
    provider.async_client.aclose.assert_awaited_once()
