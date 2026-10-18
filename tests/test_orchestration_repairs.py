"""Regressions for bounded orchestration and independent tool failures."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import TextContent

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.models.chunk import AssistantMessage, ToolCallError, ToolCallRequest, ToolMessages
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall, ToolCallList


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield await self.async_generate(**kwargs)


def request(name, **arguments):
    return ToolCallRequest(id=name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_unresolved_hidden_identity_is_an_isolated_tool_failure():
    box = LocalToolbox()
    invoked = []

    @box.tool(server_populated=["user_id"])
    async def protected(user_id: str):
        invoked.append(user_id)
        return user_id

    @box.tool(depends_on=["protected"])
    async def dependent():
        invoked.append("dependent")
        return "must not run"

    @box.tool()
    async def independent():
        await asyncio.sleep(0)
        return "ok"

    async with AsyncAIClient(providers=[ReplyProvider()], local_toolbox=box) as client:
        calls = [request("protected", user_id="forged-secret"), request("independent"), request("dependent")]
        results = await client.run_tool_calls(calls)
    assert isinstance(results[0], ToolCallError)
    assert "Missing server-populated parameters" in results[0].error
    assert "forged-secret" not in results[0].error
    assert results[1].result == "ok"
    assert isinstance(results[2], ToolCallError)
    assert "Skipped" in results[2].error
    assert invoked == []
    assert calls[0].arguments == {"user_id": "forged-secret"}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_failed_tools_return_to_model_in_chat_history(streaming):
    box = LocalToolbox()

    @box.tool()
    async def broken():
        raise RuntimeError("failed tool")

    class Provider(ReplyProvider):
        async def async_generate(self, *, chat_history, **kwargs):
            if not any(isinstance(item, ToolCallError) for item in chat_history):
                return ToolCallList(
                    model_id="test", usage=None,
                    tool_calls=[ToolCall(id="call", function_name="broken", arguments={})],
                )
            return await super().async_generate(chat_history=chat_history, **kwargs)

    async with AsyncAIClient(providers=[Provider()], local_toolbox=box) as client:
        history = []
        chat = client.stream_chat if streaming else client.chat
        events = [event async for event in chat("go", history)]
    assert isinstance(events[0], ToolMessages)
    assert isinstance(events[0][1], ToolCallError)
    assert events[0][0].id == events[0][1].tool_call_id
    assert isinstance(events[-1], AssistantMessage)
    assert events[-1].content == "done"
    assert events[-1].thinking == "reason"
    assert history[-1] is events[-1]


@pytest.mark.asyncio
async def test_single_call_still_checks_dependency_cycles():
    box = LocalToolbox()
    invoked = []

    @box.tool(depends_on=["loop"])
    async def loop():
        invoked.append(True)
        return "wrong"

    async with AsyncAIClient(providers=[ReplyProvider()], local_toolbox=box) as client:
        result = (await client.run_tool_calls([request("loop")]))[0]
    assert isinstance(result, ToolCallError)
    assert "Cyclic dependency" in result.error
    assert invoked == []


@pytest.mark.asyncio
async def test_executor_keeps_all_text_blocks_in_remote_result():
    registry = SimpleNamespace(server_params=lambda **_: [], depends_on=lambda **_: set())
    pool = SimpleNamespace(call_tool=AsyncMock(return_value=[
        SimpleNamespace(type="image"),
        TextContent(type="text", text="first"),
        TextContent(type="text", text="second"),
    ]))
    executor = Executor(pool=pool, registry=registry)
    assert (await executor.run_calls([request("multi")]))[0].result == "first\nsecond"


@pytest.mark.asyncio
async def test_registry_deduplicates_aliased_schema_and_keeps_first_route():
    primary, first_variant, duplicate_variant = LocalToolbox(), LocalToolbox(), LocalToolbox()

    @primary.tool(name="lookup")
    async def lookup(value: str):
        return value

    # Register the exact same function twice so their schemas, including title,
    # match; the primary uses a different public argument schema.
    async def variant(count: int):
        return str(count)

    first_variant.tool(name="lookup")(variant)
    duplicate_variant.tool(name="lookup")(variant)
    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=[primary, first_variant, duplicate_variant],
    ) as client:
        tools = client.list_tools()
        assert len(tools) == 2
        alias = next(t["function"]["name"] for t in tools if t["function"]["name"] != "lookup")
        assert client.pool._tool_map[alias] is first_variant
        assert (await client.run_tool_calls([request(alias, count=3)]))[0].result == "3"


@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("inf"), float("nan")])
def test_concurrency_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_concurrency"):
        AsyncAIClient(providers=[ReplyProvider()], max_concurrency=value)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_timeout_requires_a_positive_finite_number(value):
    with pytest.raises(ValueError, match="tool_timeout"):
        AsyncAIClient(providers=[ReplyProvider()], tool_timeout=value)
    with pytest.raises(ValueError, match="tool_timeout"):
        Executor(pool=None, registry=None, tool_timeout=value)


@pytest.mark.parametrize("value", [-1, True, float("inf"), float("nan")])
def test_refresh_interval_requires_a_nonnegative_finite_number(value):
    with pytest.raises(ValueError, match="refresh_interval"):
        AsyncAIClient(providers=[ReplyProvider()], refresh_interval=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
async def test_invalid_turn_limit_does_not_modify_history(streaming, limit):
    client = AsyncAIClient(providers=[ReplyProvider()])
    history = []
    chat = client.stream_chat if streaming else client.chat
    with pytest.raises(ValueError, match="max_turns"):
        await anext(chat("go", history, max_turns=limit))
    assert history == []


@pytest.mark.asyncio
async def test_concurrency_limit_is_shared_between_requests_and_released_on_cancellation():
    box = LocalToolbox()
    started = asyncio.Queue()
    active = peak = 0
    release = asyncio.Event()

    @box.tool()
    async def wait(label: str):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.put_nowait(label)
        try:
            await release.wait()
            return label
        finally:
            active -= 1

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, max_concurrency=1,
    ) as client:
        first = asyncio.create_task(client.run_tool_calls([request("wait", label="first")]))
        assert await asyncio.wait_for(started.get(), 1) == "first"
        second = asyncio.create_task(client.run_tool_calls([request("wait", label="second")]))
        await asyncio.sleep(0)
        assert started.empty()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.wait_for(started.get(), 1) == "second"
        release.set()
        assert (await asyncio.wait_for(second, 1))[0].result == "second"
    assert active == 0
    assert peak == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_or_fully_blocked_toolbox_uses_llm_only_path(streaming):
    box = LocalToolbox()

    @box.tool()
    async def hidden():
        return "hidden"

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, blocked_tools=["hidden"],
    ) as client:
        assert client.list_tools() == []
        assert not client.tools_enabled()
        chat = client.stream_chat if streaming else client.chat
        replies = [reply async for reply in chat("go", [])]
    assert replies[-1].content == "done"
    assert replies[-1].thinking == "reason"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_missing_query_is_rejected_without_changing_history(streaming):
    client = AsyncAIClient(providers=[ReplyProvider()])
    history = []
    chat = client.stream_chat if streaming else client.chat
    with pytest.raises(ValueError, match="user_query"):
        await anext(chat(None, history))
    assert history == []


@pytest.mark.asyncio
async def test_slow_async_parameter_resolution_times_out_and_releases_slot():
    box = LocalToolbox()
    cancelled = asyncio.Event()
    invoked = []

    async def resolve_user_id():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    @box.tool(server_populated=["user_id"])
    async def protected(user_id: str):
        invoked.append(user_id)
        return user_id

    @box.tool()
    async def independent():
        return "ok"

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, max_concurrency=1,
        tool_timeout=0.02, param_registry={"user_id": resolve_user_id},
    ) as client:
        results = await asyncio.wait_for(
            client.run_tool_calls([request("protected"), request("independent")]), 1,
        )
    assert isinstance(results[0], ToolCallError)
    assert "timed out after 0.02s" in results[0].error
    assert results[1].result == "ok"
    assert cancelled.is_set()
    assert invoked == []


@pytest.mark.asyncio
async def test_parameter_resolution_obeys_concurrency_limit_across_requests():
    box = LocalToolbox()
    started = asyncio.Queue()
    release = asyncio.Event()
    active = peak = 0

    async def resolve_user_id(label):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.put_nowait(label)
        try:
            await release.wait()
            return label
        finally:
            active -= 1

    @box.tool(server_populated=["user_id"])
    async def protected(user_id: str):
        return user_id

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, max_concurrency=2,
        param_registry={"user_id": resolve_user_id},
    ) as client:
        tasks = [asyncio.create_task(client.run_tool_calls(
            [request("protected")], context={"label": str(i)},
        )) for i in range(3)]
        try:
            labels = {await asyncio.wait_for(started.get(), 1) for _ in range(2)}
            assert len(labels) == 2
            await asyncio.sleep(0)
            assert started.empty()
            assert active == peak == 2
        finally:
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert [result[0].result for result in results] == ["0", "1", "2"]
    assert active == 0
    assert peak == 2


@pytest.mark.asyncio
async def test_queue_wait_does_not_consume_tool_execution_timeout():
    box = LocalToolbox()

    @box.tool()
    async def immediate():
        return "ok"

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, max_concurrency=1, tool_timeout=0.02,
    ) as client:
        async with client.executor._tool_semaphore:
            waiting = asyncio.create_task(client.run_tool_calls([request("immediate")]))
            await asyncio.sleep(0.06)
            assert not waiting.done()
        assert (await asyncio.wait_for(waiting, 1))[0].result == "ok"
