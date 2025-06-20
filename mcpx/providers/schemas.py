"""
Standardized Pydantic schemas for data exchange with LLM providers.

These models define a canonical structure for LLM responses, including
text content, tool calls, and token usage, ensuring that the AIClient
can interact with any provider in a uniform way.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ToolCall(BaseModel):
    """
    Standardized representation of a tool call requested by the LLM.

    This structure is modeled after OpenAI's format, which has become a
    de-facto standard. Providers are responsible for mapping their native
    tool-call format to this structure.
    """
    id: str
    type: str = "function"
    function: Dict[str, Any] = Field(default=..., description='e.g., {"name": "get_weather", "arguments": \'{"location": "Boston"}\'}')


class LLMUsage(BaseModel):
    """Standardized token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    """
    Standardized response object from an LLM provider call.

    This object is returned by a provider's `async_generate` method and
    is the final object yielded by the `async_stream` method.
    """
    model_id: str = Field(..., description="The actual model ID used for the response.")
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[LLMUsage] = None
    stop_reason: Optional[str] = None 