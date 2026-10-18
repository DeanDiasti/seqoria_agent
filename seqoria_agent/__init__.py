"""seqoria_agent - MCP-native AI orchestration framework.

This package provides tools for building AI applications with:
- AIClient: Synchronous client for blocking usage
- AsyncAIClient: Asynchronous client for async/await usage
- LocalToolbox: In-process tool execution without HTTP

LLM Provider Support:
- Async providers (AsyncOpenAIProvider, AsyncAnthropicProvider): For async/await usage
- Sync providers (OpenAIProvider, AnthropicProvider): For blocking/synchronous usage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seqoria_agent.client import AIClient, AsyncAIClient
from seqoria_agent.toolbox import LocalToolbox

# Always-safe provider base classes (no optional deps)
from seqoria_agent.providers import AsyncBaseProvider, BaseProvider

# Type checking imports (no runtime import cost / no optional deps required)
if TYPE_CHECKING:
    from seqoria_agent.providers import (  # noqa: F401
        AnthropicProvider,
        AsyncAnthropicProvider,
        AsyncOllamaProvider,
        AsyncOpenAIProvider,
        OllamaProvider,
        OpenAIProvider,
    )


# ---------------------------------------------------------------------------
# Lazy provider re-exports
# ---------------------------------------------------------------------------

_LAZY_EXPORTS: set[str] = {
    # HTTP providers
    "AsyncOpenAIProvider",
    "OpenAIProvider",
    "AsyncAnthropicProvider",
    "AnthropicProvider",
    "AsyncOllamaProvider",
    "OllamaProvider",
}


def __getattr__(name: str) -> Any:
    """Lazy re-export of provider classes.

    This keeps `from seqoria_agent import AsyncAIClient` lightweight while still
    supporting `from seqoria_agent import OpenAIProvider`.
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
    "LocalToolbox",
    # Base providers
    "AsyncBaseProvider",
    "BaseProvider",
    # Providers (lazy)
    "AsyncOpenAIProvider",
    "OpenAIProvider",
    "AsyncAnthropicProvider",
    "AnthropicProvider",
    "AsyncOllamaProvider",
    "OllamaProvider",
]
