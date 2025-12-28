"""seqoria_agent - MCP-native AI orchestration framework.

This package provides tools for building AI applications with:
- AIClient: Synchronous client for blocking usage
- AsyncAIClient: Asynchronous client for async/await usage
- SeqoriaMCP: Extended MCP server with Seqoria features
- LocalToolbox: In-process tool execution without HTTP

LLM Provider Support:
- Async providers (AsyncOpenAIProvider, AsyncAnthropicProvider): For async/await usage
- Sync providers (OpenAIProvider, AnthropicProvider): For blocking/synchronous usage
"""

from __future__ import annotations

from seqoria_agent.client import AIClient, AsyncAIClient
from seqoria_agent.server import SeqoriaMCP
from seqoria_agent.toolbox import LocalToolbox

# Provider imports for convenient access
from seqoria_agent.providers import (
    AsyncBaseProvider,
    BaseProvider,
    AsyncOpenAIProvider,
    OpenAIProvider,
    AsyncAnthropicProvider,
    AnthropicProvider,
)


__all__ = [
    # Core classes
    "AIClient",
    "AsyncAIClient",
    "SeqoriaMCP",
    "LocalToolbox",
    # Base providers
    "AsyncBaseProvider",
    "BaseProvider",
    # OpenAI providers
    "AsyncOpenAIProvider",
    "OpenAIProvider",
    # Anthropic providers
    "AsyncAnthropicProvider",
    "AnthropicProvider",
]
