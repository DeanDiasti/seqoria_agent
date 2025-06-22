import pytest
from seqoria_agent.client import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class DummyProvider(BaseProvider):
    def __init__(self):
        pass

    async def async_generate(self, * , chat_history, **kw):
        return LLMResponse(model_id="dummy", text="Hello world", tool_calls=[])

    async def async_stream(self, *args, **kwargs):
        async def _gen():
            yield "Hello world"
            yield LLMResponse(model_id="dummy", text="Hello world", tool_calls=[])
        return _gen()


@pytest.mark.asyncio
async def test_plain_chatbot():
    ai = AIClient(urls=[], providers=[DummyProvider()])
    async with ai as client:
        res = await client.chat("Hi", history=[])
        assert res["answer"] == "Hello world" 