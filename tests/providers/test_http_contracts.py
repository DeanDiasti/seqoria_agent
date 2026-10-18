"""Exercise provider wire contracts through HTTPX without hosted model calls."""

import asyncio
import json

import httpx
import pytest

from seqoria_agent.providers.anthropic_provider import AnthropicProvider, AsyncAnthropicProvider
from seqoria_agent.providers.errors import LLMError, MaxRetriesExceededError, QuotaExceededError
from seqoria_agent.providers.ollama_provider import AsyncOllamaProvider, OllamaProvider
from seqoria_agent.providers.openai_compatible_provider import (
    AsyncOpenAICompatibleProvider,
    OpenAICompatibleProvider,
)
from seqoria_agent.providers.openai_provider import AsyncOpenAIProvider, OpenAIProvider
from seqoria_agent.providers.schemas import ToolCallList

PROVIDERS = {
    "compatible": (OpenAICompatibleProvider, AsyncOpenAICompatibleProvider),
    "responses": (OpenAIProvider, AsyncOpenAIProvider),
    "anthropic": (AnthropicProvider, AsyncAnthropicProvider),
    "ollama": (OllamaProvider, AsyncOllamaProvider),
}
HISTORY = [{"role": "user", "content": "Look up seven records"}]
TOOLS = [
    {
        "name": "lookup",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        },
    }
]


def tool_response(kind, arguments='{"limit":7}'):
    """Native JSON response payloads for each supported provider protocol."""
    if kind == "compatible":
        return {
            "model": "test-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": arguments},
                            }
                        ],
                    },
                }
            ],
        }
    if kind == "responses":
        return {
            "model": "test-model",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "item-1",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": arguments,
                }
            ],
        }
    if kind == "anthropic":
        return {
            "model": "test-model",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "lookup",
                    "input": arguments,
                }
            ],
        }
    return {
        "model": "test-model",
        "done": True,
        "message": {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "lookup", "arguments": arguments}}
            ],
        },
    }


def stream_events(kind, arguments='{"limit":7}'):
    """Return separately framed native events, including a completion signal."""
    if kind == "compatible":
        return [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": arguments[:5],
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": arguments[5:],
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        ]
    if kind == "responses":
        return [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item-1",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "item-1",
                "delta": arguments[:5],
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "item-1",
                "delta": arguments[5:],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "item-1",
                "arguments": arguments,
            },
            # Omitted final output exercises assembly rather than replacing it with a canned result.
            {"type": "response.completed", "response": {"status": "completed", "output": []}},
        ]
    if kind == "anthropic":
        return [
            {
                "type": "message_start",
                "message": {"model": "test-model", "usage": {"input_tokens": 4}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": arguments[:5],
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": arguments[5:],
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
    return [
        {
            "model": "test-model",
            "message": {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ]
            },
            "done": False,
        },
        {"model": "test-model", "message": {}, "done": True, "done_reason": "stop"},
    ]


def wire(kind, events):
    if kind == "ollama":
        return "".join(json.dumps(event) + "\n" for event in events).encode()
    return "".join(
        "data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n\n"
        for event in events
    ).encode()


@pytest.fixture
def make_provider(monkeypatch):
    sync_client = httpx.Client
    async_client = httpx.AsyncClient

    def factory(kind, *, asynchronous=False, handler=None, max_retries=0):
        def route(request):
            if request.method == "GET":
                return httpx.Response(
                    200, json={"data": [{"id": "test-model"}], "models": [{"name": "test-model"}]}
                )
            return handler(request)

        transport = httpx.MockTransport(route)
        monkeypatch.setattr(httpx, "Client", lambda **kw: sync_client(transport=transport, **kw))
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: async_client(transport=transport, **kw)
        )
        return PROVIDERS[kind][int(asynchronous)](
            model_id="test-model",
            api_key="test",
            max_retries=max_retries,
        )

    return factory


def assert_tool_call(result):
    assert isinstance(result, ToolCallList)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call-1"
    assert call.function_name == "lookup"
    assert call.arguments == {"limit": 7}


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
def test_sync_tool_calls_preserve_arguments(make_provider, kind, streaming):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body.get("stream", False) is streaming
        assert body["model"] == "test-model"
        assert body["tools"]
        return (
            httpx.Response(200, content=wire(kind, stream_events(kind)))
            if streaming
            else httpx.Response(200, json=tool_response(kind))
        )

    with make_provider(kind, handler=handler) as provider:
        result = (
            list(provider.stream(HISTORY, tools=TOOLS))[-1]
            if streaming
            else provider.generate(HISTORY, tools=TOOLS)
        )
        assert_tool_call(result)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_tool_calls_preserve_arguments(make_provider, kind, streaming):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body.get("stream", False) is streaming
        assert body["model"] == "test-model"
        assert body["tools"]
        return (
            httpx.Response(200, content=wire(kind, stream_events(kind)))
            if streaming
            else httpx.Response(200, json=tool_response(kind))
        )

    async with make_provider(kind, asynchronous=True, handler=handler) as provider:
        result = (
            [chunk async for chunk in provider.async_stream(HISTORY, tools=TOOLS)][-1]
            if streaming
            else await provider.async_generate(HISTORY, tools=TOOLS)
        )
        assert_tool_call(result)
    assert len(requests) == 1


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("arguments", ['{"limit":', "[]", "null"])
def test_sync_invalid_tool_arguments_fail_closed(make_provider, kind, streaming, arguments):
    def handler(request):
        return (
            httpx.Response(200, content=wire(kind, stream_events(kind, arguments)))
            if streaming
            else httpx.Response(200, json=tool_response(kind, arguments))
        )

    with make_provider(kind, handler=handler) as provider:
        with pytest.raises(LLMError, match="tool arguments"):
            list(provider.stream(HISTORY, tools=TOOLS)) if streaming else provider.generate(
                HISTORY, tools=TOOLS
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("arguments", ['{"limit":', "[]", "null"])
async def test_async_invalid_tool_arguments_fail_closed(make_provider, kind, streaming, arguments):
    def handler(request):
        return (
            httpx.Response(200, content=wire(kind, stream_events(kind, arguments)))
            if streaming
            else httpx.Response(200, json=tool_response(kind, arguments))
        )

    async with make_provider(kind, asynchronous=True, handler=handler) as provider:
        with pytest.raises(LLMError, match="tool arguments"):
            if streaming:
                _ = [chunk async for chunk in provider.async_stream(HISTORY, tools=TOOLS)]
            else:
                await provider.async_generate(HISTORY, tools=TOOLS)


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("bad_event", [False, True])
def test_sync_truncated_or_malformed_streams_raise(make_provider, kind, bad_event):
    events = stream_events(kind)
    incomplete = events[:-2] if kind == "compatible" else events[:-1]
    content = (
        (b"{broken-json\n" if kind == "ollama" else b"data: {broken-json\n\n") + wire(kind, events)
        if bad_event
        else wire(kind, incomplete)
    )
    with make_provider(
        kind, handler=lambda request: httpx.Response(200, content=content)
    ) as provider:
        with pytest.raises(LLMError, match="malformed stream|without a completion"):
            list(provider.stream(HISTORY, tools=TOOLS))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("bad_event", [False, True])
async def test_async_truncated_or_malformed_streams_raise(make_provider, kind, bad_event):
    events = stream_events(kind)
    incomplete = events[:-2] if kind == "compatible" else events[:-1]
    content = (
        (b"{broken-json\n" if kind == "ollama" else b"data: {broken-json\n\n") + wire(kind, events)
        if bad_event
        else wire(kind, incomplete)
    )
    async with make_provider(
        kind, asynchronous=True, handler=lambda request: httpx.Response(200, content=content)
    ) as provider:
        with pytest.raises(LLMError, match="malformed stream|without a completion"):
            _ = [chunk async for chunk in provider.async_stream(HISTORY, tools=TOOLS)]


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("budget", [-1, 0.5, True, False, "2", None])
def test_retry_budget_rejected_before_http_client_creation(monkeypatch, kind, asynchronous, budget):
    def forbidden(**kwargs):
        pytest.fail("Invalid retry settings created an HTTP client")

    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    with pytest.raises(ValueError, match="max_retries must be a nonnegative integer"):
        PROVIDERS[kind][int(asynchronous)](
            model_id="test-model", api_key="test", max_retries=budget
        )


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
def test_sync_retries_only_before_response_output(make_provider, monkeypatch, kind, streaming):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return (
            httpx.Response(200, content=wire(kind, stream_events(kind)))
            if streaming
            else httpx.Response(200, json=tool_response(kind))
        )

    with make_provider(kind, handler=handler, max_retries=1) as provider:
        result = (
            list(provider.stream(HISTORY, tools=TOOLS))[-1]
            if streaming
            else provider.generate(HISTORY, tools=TOOLS)
        )
        assert_tool_call(result)
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_retries_only_before_response_output(
    make_provider, monkeypatch, kind, streaming
):
    async def no_backoff(seconds):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return (
            httpx.Response(200, content=wire(kind, stream_events(kind)))
            if streaming
            else httpx.Response(200, json=tool_response(kind))
        )

    async with make_provider(kind, asynchronous=True, handler=handler, max_retries=1) as provider:
        result = (
            [chunk async for chunk in provider.async_stream(HISTORY, tools=TOOLS)][-1]
            if streaming
            else await provider.async_generate(HISTORY, tools=TOOLS)
        )
        assert_tool_call(result)
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize(
    "status,error", [(429, QuotaExceededError), (503, MaxRetriesExceededError)]
)
async def test_async_http_failures_keep_error_taxonomy(make_provider, kind, status, error):
    async with make_provider(
        kind,
        asynchronous=True,
        handler=lambda request: httpx.Response(status, json={"error": {"message": "unavailable"}}),
    ) as provider:
        with pytest.raises(error):
            await provider.async_generate(HISTORY)


class BlockingStream(httpx.AsyncByteStream):
    """A response paused after its first visible delta, as on a live connection."""

    def __init__(self, first):
        self.first = first
        self.reading = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield self.first
        self.reading.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def text_delta(kind):
    return {
        "compatible": {"choices": [{"delta": {"content": "partial"}}]},
        "responses": {"type": "response.output_text.delta", "delta": "partial"},
        "anthropic": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "partial"},
        },
        "ollama": {"message": {"content": "partial"}, "done": False},
    }[kind]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", PROVIDERS)
async def test_async_cancellation_closes_response_without_replaying(make_provider, kind):
    response_stream = BlockingStream(wire(kind, [text_delta(kind)]))
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=response_stream)

    async with make_provider(kind, asynchronous=True, handler=handler, max_retries=3) as provider:
        stream = provider.async_stream(HISTORY)
        first = await anext(stream)
        assert first.text == "partial"
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(response_stream.reading.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await stream.aclose()
    assert response_stream.closed
    assert attempts == 1


def text_response(kind):
    usage = {"input_tokens": 4, "output_tokens": 5}
    return {
        "compatible": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "partial",
                        "reasoning_content": "reasoning",
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        },
        "responses": {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "partial"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "reasoning"}]},
            ],
            "usage": usage,
        },
        "anthropic": {
            "content": [
                {"type": "text", "text": "partial"},
                {"type": "thinking", "thinking": "reasoning"},
            ],
            "stop_reason": "end_turn",
            "usage": usage,
        },
        "ollama": {
            "message": {"content": "partial", "thinking": "reasoning"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 4,
            "eval_count": 5,
        },
    }[kind]


def text_events(kind):
    if kind == "compatible":
        return [
            text_delta(kind),
            {"choices": [{"delta": {"reasoning_content": "reasoning"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
            },
            "[DONE]",
        ]
    if kind == "responses":
        return [
            text_delta(kind),
            {"type": "response.reasoning_summary_text.delta", "delta": "reasoning"},
            {"type": "response.completed", "response": text_response(kind)},
        ]
    if kind == "anthropic":
        return [
            {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
            text_delta(kind),
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "reasoning"},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
    return [
        text_delta(kind),
        {"message": {"thinking": "reasoning"}, "done": False},
        {
            "message": {},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 4,
            "eval_count": 5,
        },
    ]


def assert_text(result):
    assert result.type == "llm_response"
    assert result.text == "partial"
    assert result.thinking == "reasoning"
    assert result.usage.prompt_tokens == 4
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 9


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
def test_sync_text_reasoning_usage_contract(make_provider, kind, streaming):
    def handler(request):
        return (
            httpx.Response(200, content=wire(kind, text_events(kind)))
            if streaming
            else httpx.Response(200, json=text_response(kind))
        )

    with make_provider(kind, handler=handler) as provider:
        result = list(provider.stream(HISTORY))[-1] if streaming else provider.generate(HISTORY)
        assert_text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_text_reasoning_usage_contract(make_provider, kind, streaming):
    def handler(request):
        return (
            httpx.Response(200, content=wire(kind, text_events(kind)))
            if streaming
            else httpx.Response(200, json=text_response(kind))
        )

    async with make_provider(kind, asynchronous=True, handler=handler) as provider:
        result = (
            [chunk async for chunk in provider.async_stream(HISTORY)][-1]
            if streaming
            else await provider.async_generate(HISTORY)
        )
        assert_text(result)


def parallel_response_events():
    events = []
    for index in (0, 1):
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {
                    "type": "function_call",
                    "id": f"item-{index}",
                    "call_id": f"call-{index}",
                    "name": "lookup",
                    "arguments": "",
                },
            }
        )
    for fragment in ('{"limit":', "7}"):
        for index in (0, 1):
            events.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": f"item-{index}",
                    "delta": fragment,
                }
            )
    for index in (1, 0):
        events.append({"type": "response.function_call_arguments.done", "item_id": f"item-{index}"})
    events.append({"type": "response.completed", "response": {"status": "completed", "output": []}})
    return events


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_responses_parallel_tool_arguments_are_isolated(make_provider, asynchronous):
    def handler(request):
        return httpx.Response(200, content=wire("responses", parallel_response_events()))

    provider = make_provider("responses", asynchronous=asynchronous, handler=handler)
    if asynchronous:
        async with provider:
            result = [chunk async for chunk in provider.async_stream(HISTORY, tools=TOOLS)][-1]
    else:
        with provider:
            result = list(provider.stream(HISTORY, tools=TOOLS))[-1]
    assert [call.id for call in result.tool_calls] == ["call-0", "call-1"]
    assert all(call.arguments == {"limit": 7} for call in result.tool_calls)


@pytest.mark.parametrize("completion", ["finish_reason", "done_marker"])
def test_compatible_endpoint_accepts_either_completion_signal(make_provider, completion):
    events = stream_events("compatible")
    events = events[:-1] if completion == "finish_reason" else events[:2] + ["[DONE]"]
    with make_provider(
        "compatible",
        handler=lambda request: httpx.Response(200, content=wire("compatible", events)),
    ) as provider:
        assert_tool_call(list(provider.stream(HISTORY, tools=TOOLS))[-1])


class FailingSyncStream(httpx.SyncByteStream):
    def __init__(self, first):
        self.first = first
        self.closed = False

    def __iter__(self):
        yield self.first
        raise httpx.ReadError("connection dropped")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("close_early", [False, True])
def test_sync_partial_stream_closes_without_retry(make_provider, kind, close_early):
    body = FailingSyncStream(wire(kind, [text_delta(kind)]))
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=body)

    with make_provider(kind, handler=handler, max_retries=3) as provider:
        stream = provider.stream(HISTORY)
        assert next(stream).text == "partial"
        if close_early:
            stream.close()
        else:
            with pytest.raises(LLMError, match="connection dropped"):
                next(stream)
    assert body.closed
    assert attempts == 1


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_semantically_truncated_tool_response_rejected(make_provider, kind, asynchronous):
    response = tool_response(kind)
    if kind == "compatible":
        response["choices"][0]["finish_reason"] = "length"
    elif kind == "responses":
        response["status"] = "incomplete"
    elif kind == "anthropic":
        response["stop_reason"] = "max_tokens"
    else:
        response["done_reason"] = "length"
    provider = make_provider(
        kind, asynchronous=asynchronous, handler=lambda request: httpx.Response(200, json=response)
    )
    if asynchronous:
        async with provider:
            with pytest.raises(LLMError, match="did not complete|before tool calls completed"):
                await provider.async_generate(HISTORY, tools=TOOLS)
    else:
        with (
            provider,
            pytest.raises(LLMError, match="did not complete|before tool calls completed"),
        ):
            provider.generate(HISTORY, tools=TOOLS)


def malformed_response(kind, shape):
    field = {
        "compatible": "choices",
        "responses": "output",
        "anthropic": "content",
        "ollama": "message",
    }[kind]
    if kind == "compatible":
        return {
            "choices": [{} if shape == "missing" else {"message": None if shape == "null" else []}]
        }
    return {} if shape == "missing" else {field: None if shape == "null" else "unexpected"}


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("shape", ["missing", "null", "wrong_type"])
@pytest.mark.asyncio
async def test_malformed_success_payload_is_an_error(make_provider, kind, asynchronous, shape):
    provider = make_provider(
        kind,
        asynchronous=asynchronous,
        handler=lambda request: httpx.Response(200, json=malformed_response(kind, shape)),
    )
    if asynchronous:
        async with provider:
            with pytest.raises(LLMError, match="malformed response"):
                await provider.async_generate(HISTORY)
    else:
        with provider, pytest.raises(LLMError, match="malformed response"):
            provider.generate(HISTORY)


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_malformed_success_falls_back_before_output(make_provider, kind, streaming):
    from seqoria_agent.client.ai_client.llm_interface import LLMInterface

    attempts = {"first": 0, "second": 0}

    def bad(request):
        attempts["first"] += 1
        return (
            httpx.Response(200, content=wire(kind, [{}]))
            if streaming
            else httpx.Response(200, json=malformed_response(kind, "missing"))
        )

    def good(request):
        attempts["second"] += 1
        return (
            httpx.Response(200, content=wire(kind, text_events(kind)))
            if streaming
            else httpx.Response(200, json=text_response(kind))
        )

    async with (
        make_provider(kind, asynchronous=True, handler=bad) as first,
        make_provider(kind, asynchronous=True, handler=good) as second,
    ):
        interface = LLMInterface([first, second])
        result = (
            [chunk async for chunk in interface.stream(HISTORY)][-1]
            if streaming
            else await interface.generate(HISTORY)
        )
        assert result.content == "partial"
    assert attempts == {"first": 1, "second": 1}


def empty_response(kind):
    return {
        "compatible": {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        "responses": {"status": "completed", "output": []},
        "anthropic": {"content": [], "stop_reason": "end_turn"},
        "ollama": {"message": {"content": ""}, "done": True},
    }[kind]


def empty_events(kind):
    return {
        "compatible": [{"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]"],
        "responses": [{"type": "response.completed", "response": empty_response(kind)}],
        "anthropic": [{"type": "message_start", "message": {}}, {"type": "message_stop"}],
        "ollama": [{"message": {"content": ""}, "done": True}],
    }[kind]


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_empty_text_is_still_a_valid_completion(make_provider, kind, asynchronous, streaming):
    def handler(request):
        return (
            httpx.Response(200, content=wire(kind, empty_events(kind)))
            if streaming
            else httpx.Response(200, json=empty_response(kind))
        )

    provider = make_provider(kind, asynchronous=asynchronous, handler=handler)
    if asynchronous:
        async with provider:
            result = (
                [chunk async for chunk in provider.async_stream(HISTORY)][-1]
                if streaming
                else await provider.async_generate(HISTORY)
            )
    else:
        with provider:
            result = list(provider.stream(HISTORY))[-1] if streaming else provider.generate(HISTORY)
    assert result.type == "llm_response"
    assert result.text in {None, ""}


@pytest.mark.parametrize("kind", PROVIDERS)
@pytest.mark.asyncio
async def test_malformed_stream_after_output_does_not_fall_back(make_provider, kind):
    from seqoria_agent.client.ai_client.llm_interface import LLMInterface

    second_attempts = 0
    content = wire(kind, [text_delta(kind), {}])

    def good(request):
        nonlocal second_attempts
        second_attempts += 1
        return httpx.Response(200, content=wire(kind, text_events(kind)))

    async with (
        make_provider(
            kind, asynchronous=True, handler=lambda request: httpx.Response(200, content=content)
        ) as first,
        make_provider(kind, asynchronous=True, handler=good) as second,
    ):
        stream = LLMInterface([first, second]).stream(HISTORY)
        assert (await anext(stream)).content == "partial"
        with pytest.raises(LLMError, match="malformed stream event"):
            await anext(stream)
        await stream.aclose()
    assert second_attempts == 0
