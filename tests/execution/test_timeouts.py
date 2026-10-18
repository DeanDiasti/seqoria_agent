"""Tool deadlines cover parameter resolution and release execution capacity."""

import asyncio

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_timeout_requires_a_positive_finite_number(value):
    with pytest.raises(ValueError, match="tool_timeout"):
        AsyncAIClient(providers=[ReplyProvider()], tool_timeout=value)
    with pytest.raises(ValueError, match="tool_timeout"):
        Executor(pool=None, registry=None, tool_timeout=value)


def request(name, **arguments):
    return ToolCallRequest(id=name, name=name, arguments=arguments)


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
        providers=[ReplyProvider()],
        local_toolbox=box,
        max_concurrency=1,
        tool_timeout=0.02,
        param_registry={"user_id": resolve_user_id},
    ) as client:
        results = await asyncio.wait_for(
            client.run_tool_calls([request("protected"), request("independent")]), 1
        )
    assert isinstance(results[0], ToolCallError)
    assert "timed out after 0.02s" in results[0].error
    assert results[1].result == "ok"
    assert cancelled.is_set()
    assert invoked == []


@pytest.mark.asyncio
async def test_queue_wait_does_not_consume_tool_execution_timeout():
    box = LocalToolbox()

    @box.tool()
    async def immediate():
        return "ok"

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, max_concurrency=1, tool_timeout=0.02
    ) as client:
        async with client.executor._tool_semaphore:
            waiting = asyncio.create_task(client.run_tool_calls([request("immediate")]))
            await asyncio.sleep(0.06)
            assert not waiting.done()
        assert (await asyncio.wait_for(waiting, 1))[0].result == "ok"
