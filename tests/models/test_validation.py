"""Incomplete tool messages and empty identifiers fail validation."""

import pytest

from seqoria_agent.models import ToolCallRequest


@pytest.mark.parametrize("kind", ["request", "result", "error"])
def test_incomplete_tool_messages_fail_validation(kind):
    from pydantic import ValidationError

    from seqoria_agent.models import ToolCallError, ToolCallResult

    model = {"request": ToolCallRequest, "result": ToolCallResult, "error": ToolCallError}[kind]
    with pytest.raises(ValidationError):
        model()


@pytest.mark.parametrize(
    "kind, values",
    [
        ("request", {"id": "", "name": "tool"}),
        ("request", {"id": "call", "name": ""}),
        ("result", {"tool_call_id": "", "result": "ok"}),
        ("error", {"tool_call_id": "", "error": "failed"}),
    ],
)
def test_empty_tool_identifiers_fail_validation(kind, values):
    from pydantic import ValidationError

    from seqoria_agent.models import ToolCallError, ToolCallResult

    model = {"request": ToolCallRequest, "result": ToolCallResult, "error": ToolCallError}[kind]
    with pytest.raises(ValidationError):
        model(**values)
