"""Streaming chunk value object - used by :pymeth:`seqoria_agent.client.core.AIClient.stream_chat`."""

from __future__ import annotations

from typing import Literal, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

class LLMUsage(BaseModel):
    """Usage information for an LLM call.
    """
    model_config = ConfigDict(slots=True)
    prompt_tokens: int = Field(default=None, description="The number of tokens in the prompt.")
    completion_tokens: int = Field(default=None, description="The number of tokens in the completion.")
    total_tokens: int = Field(default=None, description="The total number of tokens in the call.")


class ChatChunk(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.
    """
    model_config = ConfigDict(slots=True)
    role: Literal["assistant"] = Field(default="assistant", description="The role of the message.")
    content: str = Field(default=None, description="The content of the message.")
    model: Optional[str] = Field(default=None, description="The model used to generate the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")

class SystemMessage(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.
    """
    model_config = ConfigDict(slots=True)
    role: Literal["system"] = Field(default=None, description="The role of the message.")
    content: str = Field(default=None, description="The content of the message.")

class AssistantMessage(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.
    """
    model_config = ConfigDict(slots=True)
    role: Literal["user", "assistant", "system"] = Field(default=None, description="The role of the message.")
    content: str = Field(default=None, description="The content of the message.")
    model: Optional[str] = Field(default=None, description="The model used to generate the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")


class UserMessage(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.
    """
    model_config = ConfigDict(slots=True)
    role: Literal["user"] = Field(default=None, description="The role of the message.")
    content: str = Field(default=None, description="The content of the message.")


class ToolCallRequest(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.

    Attributes
    ----------
    id:
        The ID of the tool call.
    type:
        The type of the tool call.
    call_id:
        The ID of the tool call.
    name:
        The name of the tool call.
    arguments:
        The arguments of the tool call.
    """

    model_config = ConfigDict(slots=True)
    id: str = Field(default=None, description="The ID of the tool call.")
    role: Literal["tool_call"] = Field(default="tool_call", description="The role of the tool call.")
    type: Literal["function_call"] = Field(default="function_call", description="The type of the tool call.")
    name: str = Field(default=None, description="The name of the tool call.")
    arguments: dict = Field(default=None, description="The arguments of the tool call.")


class ToolCallRequestList(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.
    """
    model_config = ConfigDict(slots=True)
    tool_calls: List[ToolCallRequest] = Field(default=None, description="The tool calls of the message.")
    usage: LLMUsage = Field(default=None, description="The usage of the message.")

class ToolCallResult(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.

    Attributes
    ----------
    tool_call_id:
        The ID of the tool call.
    name:
        The name of the tool.
    result:
        The result of the tool call.
    type:
        The type of the message.
    """

    model_config = ConfigDict(slots=True)
    tool_call_id: str = Field(default=None, description="The ID of the tool call.")
    role: Literal["tool_result"] = Field(default="tool_result", description="The role of the message.")
    name: str = Field(default=None, description="The name of the tool.")
    result: str = Field(default=None, description="The result of the tool call.")
    type: Literal["tool_use"] = Field(default="tool_use", description="The type of the message.")


class ToolCallError(BaseModel):  # noqa: D101 – container for failed tool execution
    """Represents a *failed* tool execution.

    This mirrors :class:`ToolCallResult` so callers can treat both
    objects uniformly (they both expose a ``result`` string).  The
    :pyattr:`role` and :pyattr:`type` fields are set to ``tool_error``
    which allows consumers to distinguish successes from failures while
    preserving backwards-compatibility for code that only inspects the
    ``result`` attribute.
    """

    model_config = ConfigDict(slots=True)
    tool_call_id: str = Field(default=None, description="The ID of the tool call that failed.")
    role: Literal["tool_error"] = Field(default="tool_error", description="The role of the message.")
    name: str = Field(default=None, description="The name of the tool.")
    # Human-readable error description (e.g. timeout, circuit-breaker, …)
    error: str = Field(default=None, description="A short error description.")
    type: Literal["tool_error"] = Field(default="tool_error", description="The type of the message.")

    # Backwards-compatibility alias so code accessing ``result`` still works.
    @property
    def result(self) -> str:  # noqa: D401 – simple redirect
        return self.error


class ToolMessages(list):  # noqa: D101 – simple container
    """Container list for grouped *tool_call*/*tool_result* message pairs.

    Sub-classing :class:`list` allows callers to use ``isinstance(chunk, ToolMessages)``
    while retaining full list behaviour (indexing, iteration, etc.).  Items must be
    either :class:`ToolCallRequest` or :class:`ToolCallResult`.
    """

    def __init__(self, iterable=None):
        iterable = iterable or []
        # Validate element types for basic safety in development/debug builds.
        for item in iterable:
            if not isinstance(item, (ToolCallRequest, ToolCallResult)):
                raise TypeError("ToolMessages may only contain ToolCallRequest or ToolCallResult instances")
        super().__init__(iterable)

    def append(self, __object):  # type: ignore[override]
        if not isinstance(__object, (ToolCallRequest, ToolCallResult)):
            raise TypeError("ToolMessages may only contain ToolCallRequest or ToolCallResult instances")
        super().append(__object)

    def extend(self, __iterable):  # type: ignore[override]
        for item in __iterable:
            if not isinstance(item, (ToolCallRequest, ToolCallResult)):
                raise TypeError("ToolMessages may only contain ToolCallRequest or ToolCallResult instances")
        super().extend(__iterable)

ChatMessages = Union[SystemMessage, AssistantMessage, UserMessage, ToolCallRequest, ToolCallResult]
