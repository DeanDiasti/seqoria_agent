import pytest

from seqoria_agent.client import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class DummyProvider(BaseProvider):
    """Minimal provider that echoes a static answer."""

    async def async_generate(self, *, chat_history, tools=None, tool_choice="auto", model=None):  # noqa: ANN001
        return LLMResponse(model_id="dummy", type="llm_response", text="ok")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "delta"
            yield LLMResponse(model_id="dummy", type="llm_response", text="ok")
        return _gen()


@pytest.mark.asyncio
async def test_default_system_prompt_inserted():
    history: list[dict] = []
    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=None) as ai:
        await ai.llm_generate_once(history=history, use_tools=False)
        assert history[0]["role"] == "system"
        # The prompt should be non-empty
        assert history[0]["content"]


@pytest.mark.asyncio
async def test_custom_system_prompt():
    custom = "You are a unit-test assistant."
    history: list[dict] = []
    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=None, system_prompt=custom) as ai:
        await ai.llm_generate_once(history=history, use_tools=False)
        assert history[0]["content"] == custom


@pytest.mark.asyncio
async def test_existing_system_prompt_not_duplicated():
    existing = {"role": "system", "content": "Pre inserted"}
    history = [existing]
    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=None) as ai:
        await ai.llm_generate_once(history=history, use_tools=False)
        # Still single system prompt
        assert history[0] is existing
        assert len([msg for msg in history if msg["role"] == "system"]) == 1 