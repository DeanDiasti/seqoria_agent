import asyncio
import pytest

from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client import AIClient
from seqoria_agent.models.chunk import ToolCallRequest, ToolCallError, ToolCallResult
from seqoria_agent.providers.base_provider import BaseProvider


class DummyProvider(BaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):  # noqa: ANN001
        # Provider is irrelevant for tool execution tests
        raise RuntimeError("LLM call should not be hit in executor tests")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "dummy"
        return _gen()


@pytest.mark.asyncio
async def test_executor_returns_tool_error_on_exception():
    """A tool that raises must yield a ToolCallError with expected fields."""

    tb = LocalToolbox()

    @tb.tool()
    def failing_tool():  # noqa: D401, ANN001 – intentional failure
        raise ValueError("boom")

    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=tb) as client:
        tc = ToolCallRequest(id="fail1", name="failing_tool", arguments={})

        # Blocking helper ----------------------------------------------
        results = await client.executor.run_calls([tc])
        assert len(results) == 1
        outcome = results[0]
        assert isinstance(outcome, ToolCallError)
        assert outcome.role == "tool_error"
        assert "boom" in outcome.error

        # Streaming helper ---------------------------------------------
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
    def bad_tool():  # noqa: D401, ANN001
        raise RuntimeError("nope")

    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=tb) as client:
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