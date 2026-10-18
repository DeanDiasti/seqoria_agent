"""Provider fallback, streamed output and reasoning preservation."""

import pytest

from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.errors import LLMError
from seqoria_agent.providers.schemas import LLMResponse


class ScriptedProvider(AsyncBaseProvider):
    def __init__(self, chunks=(), *, failure=None, response=None):
        self.chunks = chunks
        self.failure = failure
        self.response = response
        self.calls = 0
        self.closed = False

    async def async_generate(self, **kwargs):
        return self.response

    async def async_stream(self, **kwargs):
        self.calls += 1
        try:
            for chunk in self.chunks:
                yield chunk
            if self.failure:
                raise self.failure
        finally:
            self.closed = True


def response(text="answer", *, kind="llm_response", thinking=None):
    return LLMResponse(model_id="test", type=kind, text=text, thinking=thinking)


@pytest.mark.asyncio
async def test_generate_preserves_thinking():
    result = await LLMInterface(
        [
            ScriptedProvider(response=response(thinking="reasoning")),
        ]
    ).generate([])
    assert isinstance(result, AssistantMessage)
    assert result.content == "answer"
    assert result.thinking == "reasoning"


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_text", [None, ""])
async def test_ignored_delta_does_not_disable_fallback(empty_text):
    first = ScriptedProvider(
        [response(empty_text, kind="llm_delta")], failure=LLMError("failed before output")
    )
    second = ScriptedProvider([response()])
    chunks = [chunk async for chunk in LLMInterface([first, second]).stream([])]
    assert len(chunks) == 1
    assert chunks[0].content == "answer"
    assert first.closed and second.closed


@pytest.mark.asyncio
async def test_empty_stream_falls_back():
    first = ScriptedProvider()
    second = ScriptedProvider([response()])
    chunks = [chunk async for chunk in LLMInterface([first, second]).stream([])]
    assert len(chunks) == 1
    assert chunks[0].content == "answer"
    assert second.calls == 1


@pytest.mark.asyncio
async def test_all_empty_streams_raise():
    with pytest.raises(RuntimeError, match="ended without a response"):
        _ = [chunk async for chunk in LLMInterface([ScriptedProvider()]).stream([])]


@pytest.mark.asyncio
async def test_combined_delta_preserves_thinking_and_does_not_replay():
    first = ScriptedProvider(
        [response("partial", kind="llm_delta", thinking="reasoning")],
        failure=LLMError("interrupted after output"),
    )
    second = ScriptedProvider([response()])
    stream = LLMInterface([first, second]).stream([])
    chunk = await anext(stream)
    assert isinstance(chunk, ChatChunk)
    assert chunk.content == "partial"
    assert chunk.thinking == "reasoning"
    with pytest.raises(LLMError, match="interrupted after output"):
        await anext(stream)
    assert first.closed
    assert second.calls == 0


@pytest.mark.asyncio
async def test_stream_closes_provider_on_consumer_exit():
    provider = ScriptedProvider([response("partial", kind="llm_delta"), response()])
    stream = LLMInterface([provider]).stream([])
    await anext(stream)
    await stream.aclose()
    assert provider.closed


@pytest.mark.asyncio
async def test_stream_construction_error_uses_fallback():
    class ConstructionFailure(ScriptedProvider):
        def async_stream(self, **kwargs):
            raise LLMError("cannot create stream")

    fallback = ScriptedProvider([response()])
    chunks = [chunk async for chunk in LLMInterface([ConstructionFailure(), fallback]).stream([])]
    assert chunks[0].content == "answer"
    assert fallback.calls == 1
