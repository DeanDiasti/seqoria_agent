"""Regressions for provider resource ownership and stream failover."""

from unittest.mock import Mock

import httpx
import pytest

from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk
from seqoria_agent.providers.anthropic_provider import AnthropicProvider
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.errors import LLMError, ModelNotFoundError
from seqoria_agent.providers.ollama_provider import OllamaProvider
from seqoria_agent.providers.openai_compatible_provider import OpenAICompatibleProvider
from seqoria_agent.providers.openai_provider import OpenAIProvider
from seqoria_agent.providers.schemas import LLMResponse


@pytest.mark.parametrize(
    ('provider_class', 'kwargs'),
    [
        (OpenAIProvider, {}),
        (OpenAICompatibleProvider, {}),
        (AnthropicProvider, {}),
        (OllamaProvider, {'validate_on_init': True}),
    ],
)
def test_failed_constructor_closes_owned_http_client(monkeypatch, provider_class, kwargs):
    """A rejected model must not leak the HTTP pool created before validation."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client)
    error = ModelNotFoundError(model_id='missing', provider='test')
    monkeypatch.setattr(provider_class, '_validate_model_sync', Mock(side_effect=error))

    with pytest.raises(ModelNotFoundError) as caught:
        provider_class(model_id='missing', api_key='test', **kwargs)

    assert caught.value is error
    assert client.is_closed


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


def response(text='answer', *, kind='llm_response', thinking=None):
    return LLMResponse(model_id='test', type=kind, text=text, thinking=thinking)


@pytest.mark.asyncio
async def test_generate_preserves_thinking():
    result = await LLMInterface([
        ScriptedProvider(response=response(thinking='reasoning')),
    ]).generate([])
    assert isinstance(result, AssistantMessage)
    assert result.content == 'answer'
    assert result.thinking == 'reasoning'


@pytest.mark.asyncio
@pytest.mark.parametrize('empty_text', [None, ''])
async def test_ignored_delta_does_not_disable_fallback(empty_text):
    first = ScriptedProvider(
        [response(empty_text, kind='llm_delta')], failure=LLMError('failed before output')
    )
    second = ScriptedProvider([response()])
    chunks = [chunk async for chunk in LLMInterface([first, second]).stream([])]
    assert len(chunks) == 1
    assert chunks[0].content == 'answer'
    assert first.closed and second.closed


@pytest.mark.asyncio
async def test_empty_stream_falls_back():
    first = ScriptedProvider()
    second = ScriptedProvider([response()])
    chunks = [chunk async for chunk in LLMInterface([first, second]).stream([])]
    assert len(chunks) == 1
    assert chunks[0].content == 'answer'
    assert second.calls == 1


@pytest.mark.asyncio
async def test_all_empty_streams_raise():
    with pytest.raises(RuntimeError, match='ended without a response'):
        _ = [chunk async for chunk in LLMInterface([ScriptedProvider()]).stream([])]


@pytest.mark.asyncio
async def test_combined_delta_preserves_thinking_and_does_not_replay():
    first = ScriptedProvider(
        [response('partial', kind='llm_delta', thinking='reasoning')],
        failure=LLMError('interrupted after output'),
    )
    second = ScriptedProvider([response()])
    stream = LLMInterface([first, second]).stream([])
    chunk = await anext(stream)
    assert isinstance(chunk, ChatChunk)
    assert chunk.content == 'partial'
    assert chunk.thinking == 'reasoning'
    with pytest.raises(LLMError, match='interrupted after output'):
        await anext(stream)
    assert first.closed
    assert second.calls == 0


@pytest.mark.asyncio
async def test_stream_closes_provider_on_consumer_exit():
    provider = ScriptedProvider([response('partial', kind='llm_delta'), response()])
    stream = LLMInterface([provider]).stream([])
    await anext(stream)
    await stream.aclose()
    assert provider.closed


@pytest.mark.asyncio
@pytest.mark.parametrize('provider_name', ['openai', 'anthropic', 'compatible', 'ollama'])
async def test_cancelling_error_body_read_closes_http_stream(monkeypatch, provider_name):
    """Cancellation between entering a response and checking status releases it."""
    import asyncio

    from seqoria_agent.providers.anthropic_provider import AsyncAnthropicProvider
    from seqoria_agent.providers.ollama_provider import AsyncOllamaProvider
    from seqoria_agent.providers.openai_compatible_provider import AsyncOpenAICompatibleProvider
    from seqoria_agent.providers.openai_provider import AsyncOpenAIProvider

    class BlockedErrorBody(httpx.AsyncByteStream):
        def __init__(self):
            self.started = asyncio.Event()
            self.closed = False

        async def __aiter__(self):
            self.started.set()
            await asyncio.Event().wait()
            yield b'error'

        async def aclose(self):
            self.closed = True

    body = BlockedErrorBody()
    client = httpx.AsyncClient(
        base_url='https://provider.test',
        transport=httpx.MockTransport(lambda request: httpx.Response(400, stream=body)),
    )
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client)
    provider = {
        'openai': AsyncOpenAIProvider,
        'anthropic': AsyncAnthropicProvider,
        'compatible': AsyncOpenAICompatibleProvider,
        'ollama': AsyncOllamaProvider,
    }[provider_name](model_id='test', api_key='test')
    stream = provider.async_stream([{'role': 'user', 'content': 'hello'}])
    task = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(body.started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
    finally:
        task.cancel()
        await stream.aclose()
        await provider.aclose()


@pytest.mark.asyncio
async def test_sync_adapter_cancellation_waits_for_pending_next_before_close():
    import asyncio
    import threading

    from seqoria_agent.client.ai_client.sync_client import _SyncProviderAdapter
    from seqoria_agent.providers.base_provider import BaseProvider

    class BlockingProvider(BaseProvider):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.closed = threading.Event()

        def generate(self, **kwargs):
            return response()

        def stream(self, **kwargs):
            try:
                self.started.set()
                self.release.wait(timeout=2)
                yield response()
            finally:
                self.closed.set()

    provider = BlockingProvider()
    stream = _SyncProviderAdapter(provider).async_stream()
    task = asyncio.create_task(anext(stream))
    try:
        assert await asyncio.to_thread(provider.started.wait, 1)
        task.cancel()
        # Let cancellation enter the adapter while next() is still running.
        await asyncio.sleep(0)
        assert not provider.closed.is_set()
        provider.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.closed.is_set()
    finally:
        provider.release.set()
        await stream.aclose()


@pytest.mark.asyncio
async def test_stream_construction_error_uses_fallback():
    class ConstructionFailure(ScriptedProvider):
        def async_stream(self, **kwargs):
            raise LLMError('cannot create stream')

    fallback = ScriptedProvider([response()])
    chunks = [chunk async for chunk in LLMInterface([ConstructionFailure(), fallback]).stream([])]
    assert chunks[0].content == 'answer'
    assert fallback.calls == 1


@pytest.mark.parametrize('serialized', [False, True])
@pytest.mark.parametrize('provider_name', ['openai', 'compatible', 'anthropic', 'ollama', 'mlx_lm', 'mlx_vlm'])
def test_failed_tool_history_preserves_error_and_call_id(serialized, provider_name):
    from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest
    from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
    from seqoria_agent.providers.mlx_lm_provider import _MLXLMProviderMixin
    from seqoria_agent.providers.mlx_vlm_provider import _MLXVLMProviderMixin
    from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
    from seqoria_agent.providers.openai_compatible_provider import _OpenAICompatibleProviderMixin
    from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

    request = ToolCallRequest(id='call_failure', name='lookup', arguments={'key': 'x'})
    failure = ToolCallError(tool_call_id='call_failure', name='lookup', error='Permission denied')
    history = [request.model_dump(), failure.model_dump()] if serialized else [request, failure]
    if provider_name == 'openai':
        _, messages = _OpenAIProviderMixin()._prepare_input_responses(history)
        assert messages[-1] == {
            'type': 'function_call_output', 'call_id': 'call_failure', 'output': 'Permission denied',
        }
    elif provider_name == 'compatible':
        messages = _OpenAICompatibleProviderMixin()._prepare_messages_chat(history)
        assert messages[-1] == {
            'role': 'tool', 'tool_call_id': 'call_failure', 'content': 'Permission denied',
        }
    elif provider_name == 'anthropic':
        _, messages = _AnthropicProviderMixin()._prepare_chat_history_sync(history)
        assert messages[-1] == {
            'role': 'user', 'content': [{
                'type': 'tool_result', 'tool_use_id': 'call_failure',
                'content': 'Permission denied', 'is_error': True,
            }],
        }
    elif provider_name == 'ollama':
        messages = _OllamaProviderMixin()._prepare_chat_history_sync(history)
        assert messages[-1] == {
            'role': 'tool', 'tool_call_id': 'call_failure',
            'content': 'Permission denied', 'tool_name': 'lookup',
        }
    else:
        mixin = _MLXLMProviderMixin() if provider_name == 'mlx_lm' else _MLXVLMProviderMixin()
        convert = mixin._prepare_chat_messages if provider_name == 'mlx_lm' else mixin._prepare_chat_messages_text_only
        messages = convert(history)
        assert messages[-1] == {
            'role': 'user',
            'content': '[tool_error] lookup (call_id=call_failure): Permission denied',
        }


@pytest.mark.parametrize('serialized', [False, True])
def test_vlm_media_conversion_preserves_later_tool_feedback(serialized, tmp_path):
    from seqoria_agent.models.chunk import ToolCallError, UserMessage
    from seqoria_agent.providers.mlx_vlm_provider import _MLXVLMProviderMixin

    user = UserMessage(content='look this up')
    failure = ToolCallError(tool_call_id='call_failure', name='lookup', error='Permission denied')
    history = [user.model_dump(), failure.model_dump()] if serialized else [user, failure]
    messages, text, images, audio = _MLXVLMProviderMixin()._last_user_message_media(history, str(tmp_path))
    assert text == 'look this up'
    assert messages[0]['content'] == text
    assert messages[-1]['content'] == '[tool_error] lookup (call_id=call_failure): Permission denied'
    assert images == [] and audio == []


@pytest.mark.asyncio
@pytest.mark.parametrize('serialized', [False, True])
async def test_http_provider_continues_after_failed_tool(monkeypatch, serialized):
    import json

    from seqoria_agent import AsyncAIClient
    from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest
    from seqoria_agent.providers.openai_compatible_provider import AsyncOpenAICompatibleProvider

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(200, json={
                'model': 'test',
                'choices': [{'message': {'tool_calls': [{
                    'id': 'call_failure', 'type': 'function',
                    'function': {'name': 'missing_tool', 'arguments': '{}'},
                }]}, 'finish_reason': 'tool_calls'}],
            })
        return httpx.Response(200, json={
            'model': 'test',
            'choices': [{'message': {'content': 'The lookup failed.'}, 'finish_reason': 'stop'}],
        })

    http_client = httpx.AsyncClient(base_url='https://provider.test', transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: http_client)
    provider = AsyncOpenAICompatibleProvider(model_id='test', api_key='test')
    history = []
    try:
        if serialized:
            # Exercise a saved conversation loaded back as plain dictionaries.
            initial = await provider.async_generate([{'role': 'user', 'content': 'lookup'}])
            call = initial.tool_calls[0]
            history = [
                {'role': 'user', 'content': 'lookup'},
                ToolCallRequest(id=call.id, name=call.function_name, arguments=call.arguments).model_dump(),
                ToolCallError(tool_call_id=call.id, name=call.function_name, error='Tool not found').model_dump(),
            ]
            result = await provider.async_generate(history)
            assert result.text == 'The lookup failed.'
        else:
            from seqoria_agent import LocalToolbox

            toolbox = LocalToolbox()

            @toolbox.tool()
            def available_tool() -> str:
                return 'available'

            async with AsyncAIClient(providers=[provider], local_toolbox=toolbox, system_prompt='') as client:
                responses = [item async for item in client.chat('lookup', history=history)]
            assert responses[-1].content == 'The lookup failed.'
            assert any(isinstance(message, ToolCallError) for message in history)
        tool_message = sent[-1]['messages'][-1]
        assert tool_message['role'] == 'tool'
        assert tool_message['tool_call_id'] == 'call_failure'
        assert tool_message['content']
    finally:
        await provider.aclose()
