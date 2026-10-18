import asyncio
import time
import pytest

from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client import AsyncAIClient
from seqoria_agent.models.chunk import ToolCallRequest
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class DummyProvider(AsyncBaseProvider):
    """Provider stub that always returns the same static response."""

    def __init__(self, *args, **kwargs):
        pass

    async def async_generate(self, *, chat_history, tools=None, tool_choice="auto", model=None):  # noqa: ANN001
        return LLMResponse(model_id="dummy", type="llm_response", text="static")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "static"
            yield LLMResponse(model_id="dummy", type="llm_response", text="static")

        return _gen()


def _build_client(tb: LocalToolbox) -> AsyncAIClient:
    """Helper to build an AsyncAIClient instance with *tb* registered."""
    return AsyncAIClient(mcp_servers=[], providers=[DummyProvider()], local_toolbox=tb)


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
            # Ensure pairing is correct
            assert req.id == res.tool_call_id
            results.append((res.name, time.perf_counter() - start))

        # Collect names in the order they arrived
        arrived_order = [name for name, _ in results]

        assert set(arrived_order) == {"slow_tool", "fast_tool"}
        # fast_tool should arrive first because slow_tool has an artificial delay
        assert arrived_order[0] == "fast_tool"

        # Ensure the blocking helper still returns the same results list
        blocking_results = await client.executor.run_calls(calls)
        assert {r.name for r in blocking_results} == {"slow_tool", "fast_tool"} 