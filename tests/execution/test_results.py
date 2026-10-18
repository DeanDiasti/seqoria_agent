"""Tool results preserve all output content and conversation identifiers."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import TextContent

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.models import ToolCallRequest, ToolCallResult
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage


class _LLMOnlyStub(AsyncBaseProvider):
    """Provider that always returns a plain LLMResponse (no tools)."""

    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kwargs):
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return LLMResponse(
            model_id="stub", type="llm_response", text="pong", usage=usage, stop_reason="end_turn"
        )

    async def async_stream(self, *, chat_history, **kwargs):
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        yield LLMResponse(
            model_id="stub", type="llm_response", text="pong", usage=usage, stop_reason="end_turn"
        )


@pytest.mark.asyncio
async def test_executor_run_calls_returns_history_entries():
    """Executor.run_calls should return a list of ToolCallResult entries."""
    toolbox = LocalToolbox()

    @toolbox.tool()
    def foo():
        return "42"

    provider = _LLMOnlyStub()
    ai = AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=toolbox)
    tc = ToolCallRequest(id="1", name="foo", arguments={})
    async with ai as client:
        entries = await client.executor.run_calls([tc])
    assert isinstance(entries, list)
    assert entries and isinstance(entries[0], ToolCallResult)
    assert entries[0].role == "tool_result"


class UnusedProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        raise AssertionError("Tools should execute without model calls")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"ai_message": "visible", "metadata": {"count": 2}, "records": [1, 2]},
        {"ai_message": {"nested": True}, "other": "preserved"},
        {"ai_message": None, "metadata": {}},
        '<result><error code="TEST">ordinary tool text</error></result>',
        '{"ai_message": "unterminated',
    ],
)
async def test_tool_output_preserves_ordinary_json_and_xml_text(payload):
    box = LocalToolbox()

    @box.tool()
    async def output():
        return payload

    direct = (await box.call_tool("output", {}))[0].text
    async with AsyncAIClient(providers=[UnusedProvider()], local_toolbox=box) as client:
        result = (await client.run_tool_calls([ToolCallRequest(id="output", name="output")]))[0]
    assert isinstance(result, ToolCallResult)
    assert result.result == direct
    if isinstance(payload, dict):
        assert json.loads(result.result) == payload
    else:
        assert result.result == payload


def request(name, **arguments):
    return ToolCallRequest(id=name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_executor_keeps_all_text_blocks_in_remote_result():
    registry = SimpleNamespace(server_params=lambda **_: [], depends_on=lambda **_: set())
    pool = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=[
                SimpleNamespace(type="image"),
                TextContent(type="text", text="first"),
                TextContent(type="text", text="second"),
            ]
        )
    )
    executor = Executor(pool=pool, registry=registry)
    assert (await executor.run_calls([request("multi")]))[0].result == "first\nsecond"
