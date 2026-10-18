"""Execution streams yield completed work early and cancel pending work on close."""

import asyncio
import time

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models.chunk import ToolCallRequest
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class _StreamProvider(AsyncBaseProvider):
    """Provider stub that always returns the same static response."""

    def __init__(self, *args, **kwargs):
        pass

    async def async_generate(self, *, chat_history, tools=None, tool_choice="auto", model=None):
        return LLMResponse(model_id="dummy", type="llm_response", text="static")

    async def async_stream(self, *args, **kwargs):
        yield LLMResponse(model_id="test", type="llm_delta", text="static")
        yield LLMResponse(model_id="dummy", type="llm_response", text="static")


def _build_client(tb: LocalToolbox) -> AsyncAIClient:
    """Helper to build an AsyncAIClient instance with *tb* registered."""
    return AsyncAIClient(mcp_servers=[], providers=[_StreamProvider()], local_toolbox=tb)


@pytest.mark.asyncio
async def test_run_call_stream_yields_results_early():
    """The fast tool should be yielded before the slow one despite order in list."""
    tb = LocalToolbox()

    @tb.tool()
    async def slow_tool():
        await asyncio.sleep(0.05)
        return "slow"

    @tb.tool()
    async def fast_tool():
        return "fast"

    async with _build_client(tb) as client:
        calls = [
            ToolCallRequest(id="call1", name="slow_tool", arguments={}),
            ToolCallRequest(id="call2", name="fast_tool", arguments={}),
        ]
        results = []
        start = time.perf_counter()
        async for req, res in client.executor.run_call_stream(tool_calls=calls):
            assert req.id == res.tool_call_id
            results.append((res.name, time.perf_counter() - start))
        arrived_order = [name for name, _ in results]
        assert set(arrived_order) == {"slow_tool", "fast_tool"}
        assert arrived_order[0] == "fast_tool"
        blocking_results = await client.executor.run_calls(calls)
        assert {r.name for r in blocking_results} == {"slow_tool", "fast_tool"}


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def _tool_request(name, call_id=None, **arguments):
    return ToolCallRequest(id=call_id or name, name=name, arguments=arguments)


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
        stream = client.run_call_stream([_tool_request("slow"), _tool_request("fast")])
        req, _ = await anext(stream)
        assert req.name == "fast"
        await stream.aclose()
        assert cancelled.is_set()
