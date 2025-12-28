import os
import sys
import types
import pytest

# -----------------------------------------------------------------------------
# Craft a fake minimal cerebras.cloud.sdk package hierarchy
# -----------------------------------------------------------------------------
root_mod = types.ModuleType("cerebras")
cloud_mod = types.ModuleType("cerebras.cloud")
sdk_mod = types.ModuleType("cerebras.cloud.sdk")

types_mod = types.ModuleType("cerebras.cloud.sdk.types")
chat_mod = types.ModuleType("cerebras.cloud.sdk.types.chat")

class DummyAsyncCerebras:  # noqa: D401
    def __init__(self, *args, **kwargs):
        pass

sdk_mod.AsyncCerebras = DummyAsyncCerebras
sdk_mod.AsyncStream = object  # placeholder

# Dummy exception classes referenced by provider
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
    setattr(sdk_mod, name, type(name, (Exception,), {}))

chat_mod.ChatCompletion = object

# Register modules into sys.modules so import machinery can find them
sys.modules["cerebras"] = root_mod
sys.modules["cerebras.cloud"] = cloud_mod
sys.modules["cerebras.cloud.sdk"] = sdk_mod
sys.modules["cerebras.cloud.sdk.types"] = types_mod
sys.modules["cerebras.cloud.sdk.types.chat"] = chat_mod

from seqoria_agent.llm_providers.cerebras_provider import CerebrasProvider  # noqa: E402
from seqoria_agent.llm_providers.schemas import ToolCall  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _set_env_var():
    os.environ.setdefault("CEREBRAS_API_KEY", "dummy")
    yield


@pytest.mark.asyncio
async def test_prepare_params_and_tool_format():
    provider = CerebrasProvider(api_key="dummy", temperature=0.3)
    params = provider._prepare_params(max_completion_tokens=42)
    assert params["max_completion_tokens"] == 42
    # ensure default from kwargs persisted
    assert params["temperature"] == 0.3

    tools = [
        {
            "function": {
                "name": "bar",
                "description": "desc",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    ]
    formatted = await provider._format_tool_call(tools)
    assert formatted[0]["function"]["name"] == "bar"

    # parse round-trip
    raw_calls = [
        {
            "id": "c1",
            "type": "function",
            "function_name": "bar",
            "arguments": "{}",
        }
    ]
    parsed = await provider._parse_tool_call(tool_call=raw_calls)
    assert isinstance(parsed[0], ToolCall) 