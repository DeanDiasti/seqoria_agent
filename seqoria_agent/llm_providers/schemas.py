"""
Standardized Pydantic schemas for data exchange with LLM providers.

These models define a canonical structure for LLM responses, including
text content, tool calls, and token usage, ensuring that the AIClient
can interact with any provider in a uniform way.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """
    Standardized representation of a tool call requested by the LLM.

    This structure is modeled after OpenAI's format, which has become a
    de-facto standard. Providers are responsible for mapping their native
    tool-call format to this structure.
    """

    id: str = Field(default=..., description="The ID of the tool call.")
    type: Literal["tool_use", "function", "function_call"] = Field(default="function", description="The type of the tool call.")
    function_name: str = Field(default=..., description="The name of the function to call.")
    arguments: Dict[str, Any] = Field(default=..., description="The arguments to pass to the function.")


class LLMUsage(BaseModel):
    """Standardized token usage information."""

    prompt_tokens: int = Field(default=..., description="The number of tokens in the prompt.")
    completion_tokens: int = Field(default=..., description="The number of tokens in the completion.")
    total_tokens: int = Field(default=..., description="The total number of tokens in the call.")


class ToolCallList(BaseModel):
    """
    Standardized representation of a list of tool calls.
    """
    model_id: str = Field(default=..., description="The actual model ID used for the response.")
    tool_calls: List[ToolCall] = Field(default=..., description="The tool calls of the message.")
    usage: Optional[LLMUsage] = Field(default=..., description="The usage of the message.")


class LLMResponse(BaseModel):
    """
    Standardized response object from an LLM provider call.

    This object is returned by a provider's `async_generate` method and
    is the final object yielded by the `async_stream` method.
    """

    model_id: Optional[str] = Field(default=..., description="The actual model ID used for the response.")
    type: Literal["llm_delta", "llm_response"] = Field(default=..., description="The type of the response.")
    text: Optional[str] = Field(default=..., description="The text of the response.")
    usage: Optional[LLMUsage] = Field(default=None, description="The usage of the response.")
    stop_reason: Optional[str] = Field(default=None, description="The reason the response stopped.")