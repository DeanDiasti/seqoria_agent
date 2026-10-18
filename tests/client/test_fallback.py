"""Provider failover stops once response content becomes visible."""

import asyncio

import pytest

from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.models.chunk import AssistantMessage
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.errors import ConnectionError
from seqoria_agent.providers.schemas import LLMResponse


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
async def test_stream_fails_over_when_iteration_fails_before_output():

    class Broken(EchoProvider):
        async def async_stream(self, **kwargs):
            raise ConnectionError("broken")
            yield

    llm = LLMInterface(providers=[Broken(), EchoProvider()])
    events = [event async for event in llm.stream([AssistantMessage(content="works")])]
    assert events[-1].content == "works"


@pytest.mark.asyncio
async def test_stream_does_not_switch_provider_after_output_is_visible():

    class Partial(EchoProvider):
        async def async_stream(self, **kwargs):
            yield LLMResponse(model_id="partial", text="first", type="llm_delta")
            raise ConnectionError("partial")

    llm = LLMInterface(providers=[Partial(), EchoProvider()])
    stream = llm.stream([AssistantMessage(content="must not appear")])
    assert (await anext(stream)).content == "first"
    with pytest.raises(ConnectionError):
        await anext(stream)
