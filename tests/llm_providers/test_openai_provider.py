import os
import sys
import types
import asyncio

import pytest

# -----------------------------------------------------------------------------
# Dynamically create a *fake* minimal openai module so the provider imports work
# -----------------------------------------------------------------------------
openai_mock = types.ModuleType("openai")

class DummyAsyncOpenAI:  # noqa: D401
    def __init__(self, *args, **kwargs):
        pass

openai_mock.AsyncOpenAI = DummyAsyncOpenAI

# Dummy exception classes referenced by the provider
for name in [
    "APIConnectionError",
    "APIStatusError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "UnprocessableEntityError",
]:
    setattr(openai_mock, name, type(name, (Exception,), {}))

# Minimal nested types for streaming generics
openai_mock.types = types.ModuleType("openai.types")
openai_mock.types.chat = types.ModuleType("openai.types.chat")
openai_mock.types.chat.ChatCompletionChunk = object  # dummy placeholder

sys.modules.setdefault("openai", openai_mock)

# -----------------------------------------------------------------------------
# Now we can safely import the provider under test
# -----------------------------------------------------------------------------
from seqoria_agent.llm_providers.openai_provider import OpenAIProvider  # noqa: E402
from seqoria_agent.models.llm_models import AssistantMessage, UserMessage  # noqa: E402
from seqoria_agent.llm_providers.schemas import ToolCall  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _set_env_var():
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    yield
    # no cleanup necessary


@pytest.mark.asyncio
async def test_prepare_params_merges_and_filters():
    provider = OpenAIProvider(api_key="dummy", temperature=0.5)
    params = provider._prepare_params(max_tokens=50, temperature=None)

    # Provided override should be present
    assert params["max_tokens"] == 50
    # None-valued override should remove the key
    assert "temperature" not in params


@pytest.mark.asyncio
async def test_prepare_chat_history_conversion():
    provider = OpenAIProvider(api_key="dummy")

    chat_history = [
        {"role": "system", "content": "sys"},
        UserMessage(role="user", content="hi"),
        AssistantMessage(role="assistant", content="hello"),
    ]

    prepared = await provider._prepare_chat_history(chat_history=chat_history)

    # Should be list of dicts with matching roles order preserved
    assert prepared[0]["role"] == "system"
    assert prepared[1]["role"] == "user"
    assert prepared[2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_parse_tool_call():
    provider = OpenAIProvider(api_key="dummy")

    raw_calls = [
        {
            "id": "1",
            "type": "function",
            "function_name": "foo",
            "arguments": "{\"x\": 1}",
        }
    ]

    result = await provider._parse_tool_call(tool_call=raw_calls)
    assert isinstance(result[0], ToolCall)
    assert result[0].arguments == {"x": 1} 