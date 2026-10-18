"""Streaming chunk value object - used by :pymeth:`seqoria_agent.AIClient.stream_chat`."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

class LLMUsage(BaseModel):
    """Usage information for an LLM call."""
    model_config = ConfigDict(slots=True)
    prompt_tokens: int = Field(default=0, description="The number of tokens in the prompt.")
    completion_tokens: int = Field(default=0, description="The number of tokens in the completion.")
    total_tokens: int = Field(default=0, description="The total number of tokens in the call.")


class ChatChunk(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response."""
    model_config = ConfigDict(slots=True)
    role: Literal["assistant"] = Field(default="assistant", description="The role of the message.")
    content: str = Field(default="", description="The content of the message.")
    thinking: Optional[str] = Field(default=None, description="Provider thinking/reasoning delta (if supported).")
    model: Optional[str] = Field(default=None, description="The model used to generate the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")

class SystemMessage(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response."""
    model_config = ConfigDict(slots=True)
    role: Literal["system"] = Field(default="system", description="The role of the message.")
    content: str = Field(default="", description="The content of the message.")

class AssistantMessage(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response."""
    model_config = ConfigDict(slots=True)
    role: Literal["user", "assistant", "system"] = Field(default="assistant", description="The role of the message.")
    content: str = Field(default="", description="The content of the message.")
    thinking: Optional[str] = Field(default=None, description="Provider thinking/reasoning (if supported).")
    model: Optional[str] = Field(default=None, description="The model used to generate the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")


class UserMessage(BaseModel):  # noqa: D101 – simple container
    """User turn: plain text or OpenAI-style multimodal content blocks (text, image, audio).

    Providers that support multimodal input (e.g. Ollama, MLX-VLM) receive ``content`` as-is
    in chat history; others may only use the text parts.
    """
    model_config = ConfigDict(slots=True)
    role: Literal["user"] = Field(default="user", description="The role of the message.")
    content: Union[str, List[Dict[str, Any]], None] = Field(
        default=None,
        description="Plain string or list of content blocks (e.g. type text, image_url, audio).",
    )


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
    """Incremental piece of a streamed chat response."""
    model_config = ConfigDict(slots=True)
    tool_calls: List[ToolCallRequest] = Field(default_factory=list, description="The tool calls of the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")

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


class ToolMessages(list):
    """Validated list of tool requests, results, and errors.

    Iterables are materialized once so generators work for construction and
    updates. Batch updates validate before changing the list.
    """

    @staticmethod
    def _validate(item):
        if not isinstance(item, (ToolCallRequest, ToolCallResult, ToolCallError)):
            raise TypeError(
                "ToolMessages may only contain ToolCallRequest, ToolCallResult, "
                "or ToolCallError instances"
            )
        return item

    def __init__(self, iterable=None):
        items = [] if iterable is None else list(iterable)
        for item in items:
            self._validate(item)
        super().__init__(items)

    def append(self, item):
        super().append(self._validate(item))

    def extend(self, iterable):
        items = list(iterable)
        for item in items:
            self._validate(item)
        super().extend(items)

    def insert(self, index, item):
        super().insert(index, self._validate(item))

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            value = list(value)
            for item in value:
                self._validate(item)
        else:
            self._validate(value)
        super().__setitem__(index, value)

    def __iadd__(self, iterable):
        self.extend(iterable)
        return self


ChatMessages = Union[
    SystemMessage, AssistantMessage, UserMessage, ToolCallRequest, ToolCallResult, ToolCallError
]
