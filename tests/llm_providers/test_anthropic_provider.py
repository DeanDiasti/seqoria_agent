import os
import sys
import types
import pytest

# -----------------------------------------------------------------------------
# Create a dummy minimal `anthropic` module so that provider imports succeed
# -----------------------------------------------------------------------------
anthropic_mock = types.ModuleType("anthropic")

class DummyAsyncAnthropic:  # noqa: D401
    def __init__(self, *args, **kwargs):
        pass

anthropic_mock.AsyncAnthropic = DummyAsyncAnthropic

for name in [
    "APIConnectionError",
    "APIStatusError",
    "AuthenticationError",
    "BadRequestError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
]:
    setattr(anthropic_mock, name, type(name, (Exception,), {}))

sys.modules.setdefault("anthropic", anthropic_mock)

from seqoria_agent.llm_providers.anthropic_provider import AnthropicProvider  # noqa: E402
from seqoria_agent.llm_providers.schemas import ToolCall  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _set_env_var():
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
    yield


@pytest.mark.asyncio
async def test_format_and_parse_tool_calls():
    provider = AnthropicProvider(api_key="dummy")

    tools = [
        {
            "function": {
                "name": "foo",
                "description": "bar",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    ]

    formatted = await provider._format_tool_call(tools)
    assert formatted[0]["name"] == "foo"

    # Round-trip through parse helper
    raw_calls = [
        {
            "id": "1",
            "type": "function",
            "function_name": "foo",
            "arguments": "{}",
        }
    ]
    parsed = await provider._parse_tool_call(tool_call=raw_calls)
    assert isinstance(parsed[0], ToolCall) 