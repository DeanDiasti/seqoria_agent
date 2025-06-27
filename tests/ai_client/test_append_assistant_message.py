from __future__ import annotations

import warnings
import pytest

from seqoria_agent.client import AIClient  # type: ignore
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import ToolCall, LLMResponse

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"seqoria_agent\.client\.core",
)


class StubProvider(BaseProvider):
    """Minimal provider that returns static text."""

    def __init__(self):  # noqa: D401 – minimal stub
        pass

    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        return LLMResponse(model_id="stub", text="pong", tool_calls=[])

    async def async_stream(self, *args, **kwargs):  # pragma: no cover – not used in this test
        raise NotImplementedError


@pytest.mark.asyncio
async def test_append_assistant_message_basic():  # noqa: D401
    ai = AIClient(urls=[], providers=[StubProvider()])
    async with ai as client:
        history: list[dict] = []
        client.append_assistant_message("hello", history=history)

        assert len(history) == 1
        entry = history[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == "hello"
        assert "tool_calls" not in entry


@pytest.mark.asyncio
async def test_append_assistant_message_with_tool_calls():  # noqa: D401
    ai = AIClient(urls=[], providers=[StubProvider()])
    async with ai as client:
        history: list[dict] = []
        tc = ToolCall(id="1", function={"name": "noop", "arguments": {}})
        client.append_assistant_message("done", history=history, tool_calls=[tc])

        assert len(history) == 1
        entry = history[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == "done"
        assert entry["tool_calls"] == [tc.model_dump()] 