import pytest

from seqoria_agent.client import AIClient  # type: ignore
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class TrackingProvider(BaseProvider):
    """Provider that records parameters passed to async_generate for assertions."""

    def __init__(self):
        self.last_tools = None
        self.last_tool_choice = None

    async def async_generate(self, *, chat_history, tools=None, tool_choice=None, **kwargs):  # type: ignore[override]
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        return LLMResponse(model_id="stub", text="pong", tool_calls=[])

    async def async_stream(self, *args, **kwargs):  # pragma: no cover – not used here
        raise NotImplementedError


@pytest.mark.asyncio
async def test_llm_generate_once_llm_only():  # noqa: D401
    provider = TrackingProvider()
    ai = AIClient(urls=[], providers=[provider])
    async with ai as client:
        hist = client.prepare_history("ping", [])
        resp = await client.llm_generate_once(hist, use_tools=False)
        assert resp.text == "pong"
        # Provider must have been called without tool args
        assert provider.last_tools is None
        assert provider.last_tool_choice is None


@pytest.mark.asyncio
async def test_llm_generate_once_with_tools():  # noqa: D401
    provider = TrackingProvider()

    # Add a dummy local tool so tools_enabled() returns True
    from seqoria_agent.toolbox import LocalToolbox

    box = LocalToolbox()

    @box.tool()
    def foo():  # noqa: D401, ANN001
        return 42

    ai = AIClient(urls=[], providers=[provider], local_toolbox=box)
    async with ai as client:
        hist = client.prepare_history("ping", [])
        resp = await client.llm_generate_once(hist, use_tools=True)
        assert resp.text == "pong"
        # Provider should have received tool metadata
        assert provider.last_tools is not None
        assert provider.last_tool_choice == "auto" 