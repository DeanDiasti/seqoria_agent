"""Provider conversion of conversation and failed tool history."""

import base64

import httpx
import pytest


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("provider_name", ["openai", "compatible", "anthropic", "ollama"])
def test_failed_tool_history_preserves_error_and_call_id(serialized, provider_name):
    from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest
    from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
    from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
    from seqoria_agent.providers.openai_compatible_provider import _OpenAICompatibleProviderMixin
    from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

    request = ToolCallRequest(id="call_failure", name="lookup", arguments={"key": "x"})
    failure = ToolCallError(tool_call_id="call_failure", name="lookup", error="Permission denied")
    history = [request.model_dump(), failure.model_dump()] if serialized else [request, failure]
    if provider_name == "openai":
        _, messages = _OpenAIProviderMixin()._prepare_input_responses(history)
        assert messages[-1] == {
            "type": "function_call_output",
            "call_id": "call_failure",
            "output": "Permission denied",
        }
    elif provider_name == "compatible":
        messages = _OpenAICompatibleProviderMixin()._prepare_messages_chat(history)
        assert messages[-1] == {
            "role": "tool",
            "tool_call_id": "call_failure",
            "content": "Permission denied",
        }
    elif provider_name == "anthropic":
        _, messages = _AnthropicProviderMixin()._prepare_chat_history_sync(history)
        assert messages[-1] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_failure",
                    "content": "Permission denied",
                    "is_error": True,
                }
            ],
        }
    elif provider_name == "ollama":
        messages = _OllamaProviderMixin()._prepare_chat_history_sync(history)
        assert messages[-1] == {
            "role": "tool",
            "tool_call_id": "call_failure",
            "content": "Permission denied",
            "tool_name": "lookup",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("serialized", [False, True])
async def test_http_provider_continues_after_failed_tool(monkeypatch, serialized):
    import json

    from seqoria_agent import AsyncAIClient
    from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest
    from seqoria_agent.providers.openai_compatible_provider import AsyncOpenAICompatibleProvider

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(
                200,
                json={
                    "model": "test",
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_failure",
                                        "type": "function",
                                        "function": {"name": "missing_tool", "arguments": "{}"},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "test",
                "choices": [
                    {"message": {"content": "The lookup failed."}, "finish_reason": "stop"}
                ],
            },
        )

    http_client = httpx.AsyncClient(
        base_url="https://provider.test", transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: http_client)
    provider = AsyncOpenAICompatibleProvider(model_id="test", api_key="test")
    history = []
    try:
        if serialized:
            # Exercise a saved conversation loaded back as plain dictionaries.
            initial = await provider.async_generate([{"role": "user", "content": "lookup"}])
            call = initial.tool_calls[0]
            history = [
                {"role": "user", "content": "lookup"},
                ToolCallRequest(
                    id=call.id, name=call.function_name, arguments=call.arguments
                ).model_dump(),
                ToolCallError(
                    tool_call_id=call.id, name=call.function_name, error="Tool not found"
                ).model_dump(),
            ]
            result = await provider.async_generate(history)
            assert result.text == "The lookup failed."
        else:
            from seqoria_agent import LocalToolbox

            toolbox = LocalToolbox()

            @toolbox.tool()
            def available_tool() -> str:
                return "available"

            async with AsyncAIClient(
                providers=[provider], local_toolbox=toolbox, system_prompt=""
            ) as client:
                responses = [item async for item in client.chat("lookup", history=history)]
            assert responses[-1].content == "The lookup failed."
            assert any(isinstance(message, ToolCallError) for message in history)
        tool_message = sent[-1]["messages"][-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_failure"
        assert tool_message["content"]
    finally:
        await provider.aclose()


class TestOllamaChatHistoryPreparation:
    """Tests for Ollama chat history preparation."""

    def test_prepare_simple_messages(self):
        """Simple text messages should be prepared correctly."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = mixin._prepare_chat_history_sync(history)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there!"

    def test_prepare_system_message(self):
        """System messages should be prepared correctly."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        history = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result = mixin._prepare_chat_history_sync(history)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_prepare_message_with_images(self):
        """Messages with images should include images field."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        test_base64 = base64.b64encode(b"fake_image").decode()
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image", "base64": test_base64},
                ],
            }
        ]

        result = mixin._prepare_chat_history_sync(history)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "What's this?" in result[0]["content"]
        assert "images" in result[0]
        assert len(result[0]["images"]) == 1

    def test_prepare_tool_call_request(self):
        """ToolCallRequest models should be converted to assistant messages."""
        from seqoria_agent.models.chunk import ToolCallRequest
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        history = [ToolCallRequest(id="1", name="get_weather", arguments={"city": "NYC"})]

        result = mixin._prepare_chat_history_sync(history)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_prepare_tool_result(self):
        """ToolCallResult models should be converted to tool messages."""
        from seqoria_agent.models.chunk import ToolCallResult
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        history = [ToolCallResult(tool_call_id="1", result="Sunny, 72°F")]

        result = mixin._prepare_chat_history_sync(history)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Sunny, 72°F"
