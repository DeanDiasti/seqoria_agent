"""Tool exceptions become isolated, correctly paired error messages."""

import asyncio

import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest, ToolCallResult
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse
from seqoria_agent.toolbox import LocalToolbox


class _ErrorProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        raise RuntimeError("LLM call should not be hit in executor tests")

    async def async_stream(self, *args, **kwargs):
        yield LLMResponse(model_id="test", type="llm_delta", text="dummy")


@pytest.mark.asyncio
async def test_executor_returns_tool_error_on_exception():
    """A tool that raises must yield a ToolCallError with expected fields."""
    tb = LocalToolbox()

    @tb.tool()
    def failing_tool():
        raise ValueError("boom")

    async with AsyncAIClient(
        mcp_servers=[], providers=[_ErrorProvider()], local_toolbox=tb
    ) as client:
        tc = ToolCallRequest(id="fail1", name="failing_tool", arguments={})
        results = await client.executor.run_calls([tc])
        assert len(results) == 1
        outcome = results[0]
        assert isinstance(outcome, ToolCallError)
        assert outcome.role == "tool_error"
        assert "boom" in outcome.error
        async for _req, res in client.executor.run_call_stream([tc]):
            assert isinstance(res, ToolCallError)
            break


@pytest.mark.asyncio
async def test_executor_stream_mixed_success_and_failure():
    """When multiple tools run, successes and failures should intermix without cancellation."""
    tb = LocalToolbox()

    @tb.tool()
    async def ok_tool():
        await asyncio.sleep(0.01)
        return "ok"

    @tb.tool()
    def bad_tool():
        raise RuntimeError("nope")

    async with AsyncAIClient(
        mcp_servers=[], providers=[_ErrorProvider()], local_toolbox=tb
    ) as client:
        calls = [
            ToolCallRequest(id="1", name="ok_tool", arguments={}),
            ToolCallRequest(id="2", name="bad_tool", arguments={}),
        ]
        seen_success = seen_error = False
        async for _req, res in client.executor.run_call_stream(calls):
            if isinstance(res, ToolCallResult):
                seen_success = True
            if isinstance(res, ToolCallError):
                seen_error = True
        assert seen_success and seen_error
