"""Regressions for message round trips and settings isolation."""


import pytest

from seqoria_agent.models import (
    AssistantMessage, LLMUsage, SystemMessage, ToolCallError, ToolCallRequest,
    ToolCallRequestList, ToolMessages, UserMessage,
)


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
    messages = ToolMessages(item for item in [request])
    messages.extend(item for item in [error])
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


def test_settings_copy_detaches_configuration_containers():
    from seqoria_agent.client.settings import ClientSettings
    from seqoria_agent import LocalToolbox

    toolbox = LocalToolbox()
    settings = ClientSettings(
        mcp_servers=["https://example.invalid/mcp?secret=credential"],
        local_toolbox=[toolbox],
        retry_policy={"max_attempts": 2},
    )
    copied = settings.copy()
    copied.mcp_servers.clear()
    copied.local_toolbox.clear()
    copied.retry_policy["max_attempts"] = 3
    assert settings.mcp_servers == ["https://example.invalid/mcp?secret=credential"]
    assert settings.local_toolbox == [toolbox]
    assert settings.retry_policy == {"max_attempts": 2}
    assert "credential" not in repr(settings)
