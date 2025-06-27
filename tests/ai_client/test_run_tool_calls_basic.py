import pytest

from seqoria_agent.client import AIClient  # type: ignore
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.schemas import ToolCall
from seqoria_agent.providers.base_provider import BaseProvider


class StubProvider(BaseProvider):
    """Minimal provider that is *not* used during the tool execution tests."""

    def __init__(self):
        pass

    async def async_generate(self, *args, **kwargs):  # pragma: no cover – not used
        raise NotImplementedError

    async def async_stream(self, *args, **kwargs):  # pragma: no cover – not used
        raise NotImplementedError


@pytest.mark.asyncio
async def test_run_tool_calls_forwarding():  # noqa: D401
    """`AIClient.run_tool_calls` must forward to `Executor.run_calls` unchanged."""

    # Create a dummy local toolbox with a simple tool so `tools_enabled` is True.
    box = LocalToolbox()

    @box.tool()
    def ping():  # noqa: D401, ANN001
        return "pong"

    client = AIClient(urls=[], providers=[StubProvider()], local_toolbox=box)
    async with client as c:
        hist: list[dict] = []
        tool_calls = [
            ToolCall(id="1", function={"name": "ping", "arguments": {}}),
        ]

        # Direct executor call
        direct = await c.executor.run_calls(tool_calls=tool_calls, history=hist)

        # Wrapper call – should be identical
        wrapped = await c.run_tool_calls(tool_calls=tool_calls, history=hist)
        assert wrapped == direct 