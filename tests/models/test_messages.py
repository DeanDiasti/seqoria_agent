"""Message containers preserve defaults, identifiers and atomic updates."""

import pytest

from seqoria_agent.models import (
    AssistantMessage,
    LLMUsage,
    SystemMessage,
    ToolCallError,
    ToolCallRequest,
    ToolCallRequestList,
    ToolMessages,
    UserMessage,
)
from seqoria_agent.models.chunk import ChatChunk


def test_message_defaults_round_trip():
    for model in (AssistantMessage, SystemMessage, UserMessage, LLMUsage, ToolCallRequestList):
        message = model()
        assert model.model_validate(message.model_dump()) == message
    assert UserMessage().role == "user"
    assert SystemMessage().role == "system"
    assert AssistantMessage().role == "assistant"
    assert LLMUsage().total_tokens == 0


def test_tool_messages_preserve_generators_and_errors():
    request = ToolCallRequest(id="call", name="example", arguments={})
    error = ToolCallError(tool_call_id="call", name="example", error="failed")
    messages = ToolMessages((item for item in [request]))
    messages.extend((item for item in [error]))
    assert messages == [request, error]
    messages += (item for item in [request])
    assert messages == [request, error, request]


def test_tool_messages_validate_updates_atomically():
    error = ToolCallError(tool_call_id="call", name="example", error="failed")
    messages = ToolMessages([error])
    for update in (
        lambda: messages.extend(iter([error, "invalid"])),
        lambda: messages.insert(0, "invalid"),
        lambda: messages.__setitem__(0, "invalid"),
        lambda: messages.__setitem__(slice(None), iter([error, "invalid"])),
    ):
        with pytest.raises(TypeError):
            update()
        assert messages == [error]


def test_chat_chunk_dataclass():
    """Ensure ChatChunk model round-trips basic fields after refactor."""
    chunk = ChatChunk(role="assistant", content="hi", model="stub")
    assert chunk.role == "assistant"
    assert chunk.content == "hi"
    assert chunk.model == "stub"
    assert chunk.usage is None


def test_valid_tool_messages_round_trip_and_argument_defaults_are_independent():
    from seqoria_agent.models import ToolCallError, ToolCallResult

    first = ToolCallRequest(id="one", name="tool")
    second = ToolCallRequest(id="two", name="tool")
    first.arguments["value"] = 1
    assert second.arguments == {}
    for message in [
        first,
        second,
        ToolCallResult(tool_call_id="one", result="ok"),
        ToolCallError(tool_call_id="two", error="failed"),
    ]:
        assert type(message).model_validate(message.model_dump()) == message
