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

from typing import TYPE_CHECKING, Any

from seqoria_agent.client import AIClient, AsyncAIClient
from seqoria_agent.server import SeqoriaMCP
from seqoria_agent.toolbox import LocalToolbox

# Always-safe provider base classes (no optional deps)
from seqoria_agent.providers import AsyncBaseProvider, BaseProvider

# Type checking imports (no runtime import cost / no optional deps required)
if TYPE_CHECKING:
    from seqoria_agent.providers import (  # noqa: F401
        AnthropicProvider,
        AsyncAnthropicProvider,
        AsyncMLXLMProvider,
        AsyncMLXVLMProvider,
        AsyncOllamaProvider,
        AsyncOpenAIProvider,
        MLXLMProvider,
        MLXVLMProvider,
        OllamaProvider,
        OpenAIProvider,
    )


# ---------------------------------------------------------------------------
# Lazy provider re-exports
# ---------------------------------------------------------------------------

_LAZY_EXPORTS: set[str] = {
    # Providers (optional deps)
    "AsyncOpenAIProvider",
    "OpenAIProvider",
    "AsyncAnthropicProvider",
    "AnthropicProvider",
    "AsyncOllamaProvider",
    "OllamaProvider",
    "AsyncMLXLMProvider",
    "MLXLMProvider",
    "AsyncMLXVLMProvider",
    "MLXVLMProvider",
}


def __getattr__(name: str) -> Any:
    """Lazy re-export of optional provider classes.

    This keeps `from seqoria_agent import AsyncAIClient` lightweight while still
    supporting `from seqoria_agent import OpenAIProvider` when the extra is installed.
    """
    if name in _LAZY_EXPORTS:
        from seqoria_agent import providers as _providers

        return getattr(_providers, name)
    raise AttributeError(f"module 'seqoria_agent' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS))


__all__ = [
    # Core classes
    "AIClient",
    "AsyncAIClient",
    "SeqoriaMCP",
    "LocalToolbox",
    # Base providers
    "AsyncBaseProvider",
    "BaseProvider",
    # Optional providers (lazy)
    "AsyncOpenAIProvider",
    "OpenAIProvider",
    "AsyncAnthropicProvider",
    "AnthropicProvider",
    "AsyncOllamaProvider",
    "OllamaProvider",
    "AsyncMLXLMProvider",
    "MLXLMProvider",
    "AsyncMLXVLMProvider",
    "MLXVLMProvider",
]
