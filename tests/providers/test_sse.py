"""Compatible streaming honors SSE framing before decoding its JSON payloads."""

import json

import httpx
import pytest

from seqoria_agent.providers._sse import aiter_sse_data, iter_sse_data
from seqoria_agent.providers.errors import LLMError
from seqoria_agent.providers.openai_compatible_provider import (
    AsyncOpenAICompatibleProvider,
    OpenAICompatibleProvider,
)
from seqoria_agent.providers.schemas import ToolCallList

HISTORY = [{"role": "user", "content": "Look up seven records"}]


def frame(payload):
    """Encode pretty JSON as multiple SSE data fields with ignorable metadata."""
    return (
        "event: message\nid: event-1\nretry: 1000\n: keep-alive\n"
        + "\n".join("data: " + line for line in json.dumps(payload, indent=2).splitlines())
        + "\n\n"
    )


def tool_stream(*, done=True):
    events = [
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
                                    "arguments": '{"limit":',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "7}"}}]}}]},
    ]
    data = ": comment-only heartbeat\n\nevent: ping\nid: ping-1\n\ndata:\n\n" + "".join(
        frame(event) for event in events
    )
    if done:
        # The provider accepts the standard marker even without a finish_reason.
        data += "data:[DONE]\n\n"
    return data


@pytest.fixture
def provider_factory(monkeypatch):
    sync_client = httpx.Client
    async_client = httpx.AsyncClient

    def make(content, asynchronous):
        def route(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "test-model"}]})
            return httpx.Response(200, content=content)

        transport = httpx.MockTransport(route)
        monkeypatch.setattr(httpx, "Client", lambda **kw: sync_client(transport=transport, **kw))
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: async_client(transport=transport, **kw)
        )
        provider_class = AsyncOpenAICompatibleProvider if asynchronous else OpenAICompatibleProvider
        return provider_class(model_id="test-model", api_key="offline-test", max_retries=0)

    return make


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_multiline_tool_events_keep_chunked_arguments(provider_factory, asynchronous):
    provider = provider_factory(tool_stream(), asynchronous)
    if asynchronous:
        async with provider:
            result = [chunk async for chunk in provider.async_stream(HISTORY)][-1]
    else:
        with provider:
            result = list(provider.stream(HISTORY))[-1]
    assert isinstance(result, ToolCallList)
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].function_name == "lookup"
    assert result.tool_calls[0].arguments == {"limit": 7}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["malformed", "truncated"])
@pytest.mark.asyncio
async def test_multiline_malformed_or_truncated_stream_fails(
    provider_factory, asynchronous, failure
):
    data = (
        'data: {"choices":\ndata: broken-json}\n\ndata: [DONE]\n\n'
        if failure == "malformed"
        else tool_stream(done=False)
    )
    provider = provider_factory(data, asynchronous)
    if asynchronous:
        async with provider:
            with pytest.raises(
                LLMError, match="malformed stream event JSON|without a completion event"
            ):
                _ = [chunk async for chunk in provider.async_stream(HISTORY)]
    else:
        with (
            provider,
            pytest.raises(LLMError, match="malformed stream event JSON|without a completion event"),
        ):
            list(provider.stream(HISTORY))


@pytest.mark.parametrize(
    "lines,expected",
    [
        (
            [
                ": ping",
                "event: ping",
                "id: 1",
                "retry: 1000",
                "",
                "data: first",
                "data: second",
                "",
            ],
            ["first\nsecond"],
        ),
        (["data:first", ": comment within an event", "data: second"], ["first\nsecond"]),
        (["data", "", "data:", "", "data: [DONE]"], ["", "", "[DONE]"]),
        (["event: message", "id: 1", "unknown: ignored", ""], []),
        (["data:  leading-space", ""], [" leading-space"]),
    ],
)
@pytest.mark.asyncio
async def test_sync_and_async_framing_agree(lines, expected):
    async def async_lines():
        for line in lines:
            yield line

    assert list(iter_sse_data(lines)) == expected
    assert [data async for data in aiter_sse_data(async_lines())] == expected
