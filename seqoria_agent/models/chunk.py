"""Streaming chunk value object - used by :pymeth:`seqoria_agent.AIClient.stream_chat`."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

class LLMUsage(BaseModel):
    """Usage information for an LLM call."""
    prompt_tokens: int = Field(default=0, description="The number of tokens in the prompt.")
    completion_tokens: int = Field(default=0, description="The number of tokens in the completion.")
    total_tokens: int = Field(default=0, description="The total number of tokens in the call.")


class ChatChunk(BaseModel):  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response."""
    role: Literal["assistant"] = Field(default="assistant", description="The role of the message.")
    content: str = Field(default="", description="The content of the message.")
    thinking: Optional[str] = Field(default=None, description="Provider thinking/reasoning delta (if supported).")
    model: Optional[str] = Field(default=None, description="The model used to generate the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")

class SystemMessage(BaseModel):  # noqa: D101 – simple container
    """System instructions included in conversation history."""
    role: Literal["system"] = Field(default="system", description="The role of the message.")
    content: str = Field(default="", description="The content of the message.")

class AssistantMessage(BaseModel):  # noqa: D101 – simple container
    """Complete assistant response with optional reasoning and token usage."""
    role: Literal["user", "assistant", "system"] = Field(default="assistant", description="The role of the message.")
    content: str = Field(default="", description="The content of the message.")
    thinking: Optional[str] = Field(default=None, description="Provider thinking/reasoning (if supported).")
    model: Optional[str] = Field(default=None, description="The model used to generate the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")


class UserMessage(BaseModel):  # noqa: D101 – simple container
    """User turn: plain text or OpenAI-style multimodal content blocks (text, image, audio).

    Providers that support multimodal input (e.g. Ollama) receive ``content`` as-is
    in chat history; others may only use the text parts.
    """
    role: Literal["user"] = Field(default="user", description="The role of the message.")
    content: Union[str, List[Dict[str, Any]], None] = Field(
        default=None,
        description="Plain string or list of content blocks (e.g. type text, image_url, audio).",
    )


class ToolCallRequest(BaseModel):  # noqa: D101 – simple container
    """A model request to execute a named tool.

    Attributes
    ----------
    id:
        The ID of the tool call.
    type:
        The type of the tool call.
    name:
        The name of the tool call.
    arguments:
        The arguments of the tool call.
    """

    id: str = Field(min_length=1, description="The ID of the tool call.")
    role: Literal["tool_call"] = Field(default="tool_call", description="The role of the tool call.")
    type: Literal["function_call"] = Field(default="function_call", description="The type of the tool call.")
    name: str = Field(min_length=1, description="The name of the tool call.")
    arguments: dict = Field(default_factory=dict, description="The arguments of the tool call.")


class ToolCallRequestList(BaseModel):  # noqa: D101 – simple container
    """A batch of model-requested tool calls with optional usage."""
    tool_calls: List[ToolCallRequest] = Field(default_factory=list, description="The tool calls of the message.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the message.")

class ToolCallResult(BaseModel):  # noqa: D101 – simple container
    """A successful tool execution result, paired with its request by call ID.

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

    tool_call_id: str = Field(min_length=1, description="The ID of the tool call.")
    role: Literal["tool_result"] = Field(default="tool_result", description="The role of the message.")
    name: Optional[str] = Field(default=None, description="The name of the tool.")
    result: str = Field(description="The result of the tool call.")
    type: Literal["tool_use"] = Field(default="tool_use", description="The type of the message.")


class ToolCallError(BaseModel):  # noqa: D101 – container for failed tool execution
    """Represents a *failed* tool execution.

    Errors use an explicit ``error`` payload and ``tool_error`` role/type.
    Inspect the message type before reading a success result or error.
    """

    tool_call_id: str = Field(min_length=1, description="The ID of the tool call that failed.")
    role: Literal["tool_error"] = Field(default="tool_error", description="The role of the message.")
    name: Optional[str] = Field(default=None, description="The name of the tool.")
    # Human-readable error description (e.g. timeout, circuit-breaker, …)
    error: str = Field(min_length=1, description="A short error description.")
    type: Literal["tool_error"] = Field(default="tool_error", description="The type of the message.")



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
