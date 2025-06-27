import pytest

from seqoria_agent.client import AIClient  # type: ignore
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class StubProvider(BaseProvider):
    """Minimal provider that returns static responses for testing."""

    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        return LLMResponse(model_id="stub", text="pong", tool_calls=[])

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        async def _gen():
            yield "p"
            yield "o"
            yield "n"
            yield "g"
            yield LLMResponse(model_id="stub", text="pong", tool_calls=[])

        return _gen()


@pytest.mark.asyncio
async def test_prepare_history_adds_system_and_user():  # noqa: D401
    ai = AIClient(urls=[], providers=[StubProvider()])
    async with ai as client:
        history: list[dict] = []
        out = client.prepare_history("hello", history)

        # Returns same object
        assert out is history

        # System prompt + user message
        assert len(history) == 2
        assert history[0]["role"] == "system"
        assert history[1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_prepare_history_preserves_existing_system():  # noqa: D401
    ai = AIClient(urls=[], providers=[StubProvider()])
    async with ai as client:
        history = [{"role": "system", "content": "Hi"}]
        client.prepare_history("yo", history)

        # System not duplicated
        assert len(history) == 2
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "yo"


@pytest.mark.asyncio
async def test_tools_enabled_toggle():  # noqa: D401
    provider = StubProvider()
    # No tools (LLM-only)
    ai = AIClient(urls=[], providers=[provider])
    async with ai as client:
        assert client.tools_enabled() is False

    # With local toolbox – now tools available
    box = LocalToolbox()

    @box.tool()
    def foo():  # noqa: D401, ANN001
        return 1

    ai2 = AIClient(urls=[], providers=[provider], local_toolbox=box)
    async with ai2 as client:
        assert client.tools_enabled() is True 