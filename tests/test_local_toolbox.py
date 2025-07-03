import asyncio
import pytest

from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class DummyProvider(BaseProvider):
    """Bare-bones provider that just echoes a static response."""

    def __init__(self, *args, **kwargs):
        pass

    async def async_generate(self, *, chat_history, tools, tool_choice="auto"):
        return LLMResponse(model_id="dummy", type="llm_response", text="Static answer")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "Static answer"
            yield LLMResponse(model_id="dummy", type="llm_response", text="Static answer")

        return _gen()


def build_ai_client(toolbox: LocalToolbox):
    provider = DummyProvider()
    return AIClient(urls=[], providers=[provider], local_toolbox=toolbox)


@pytest.mark.asyncio
async def test_local_tool_execution():
    tb = LocalToolbox()

    @tb.tool()
    def add(a: int, b: int) -> int:
        return int(a) + int(b)

    async with build_ai_client(tb) as client:
        result = await client.call_mcp_tool("add", {"a": 2, "b": 3})
        xml = result[0].text
        assert "5" in xml


@pytest.mark.asyncio
async def test_inline_display_tool():
    from seqoria_agent.models.envelope import DisplayEnvelope

    tb = LocalToolbox()

    @tb.display_tool()
    def greet(name: str, show_user: bool = False):
        if show_user:
            return DisplayEnvelope(type="markdown", payload=f"Hello {name}")
        return f"Hi {name}"

    async with build_ai_client(tb) as client:
        # Invoke with show_user flag to trigger display path
        res = await client.call_mcp_tool("greet", {"name": "Bob", "show_user": True})
        assert "<display>" in res[0].text 