import pytest

from seqoria_agent.client import AsyncAIClient  # type: ignore
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.base_provider import AsyncBaseProvider

class StubProvider(AsyncBaseProvider):
    """Provider that does nothing but satisfy the interface for tests."""

    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        # Never called in the remaining tests
        return None

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        async def _gen():
            yield ""

        return _gen()


# ---------------------------------------------------------------------------
# Remaining public behaviour – tools_enabled()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_enabled_toggle():  # noqa: D401
    provider = StubProvider()
    # No tools (LLM-only)
    ai = AsyncAIClient(mcp_servers=[], providers=[provider])
    async with ai as client:
        assert client.tools_enabled() is False

    # With local toolbox – now tools available
    box = LocalToolbox()

    @box.tool()
    def foo():  # noqa: D401, ANN001
        return 1

    ai2 = AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=box)
    async with ai2 as client:
        assert client.tools_enabled() is True 