"""
seqoria_agent.providers: Package for managing LLM provider integrations.

This package provides modules for interacting with different
Large Language Model (LLM) providers, such as OpenAI, Anthropic,
and others. It defines a standardized interface for querying
these providers and handles common tasks like authentication,
error handling, and response formatting.

Provider Classes
----------------
Each provider comes in two variants:
- Async* (e.g., AsyncOpenAIProvider): For asynchronous operations using async/await
- Sync (e.g., OpenAIProvider): For synchronous/blocking operations

The async variants are used by AIClient and the agent framework, while sync
variants are useful for simple scripts, notebooks, and non-async contexts.

Installation
------------
Providers in this package use HTTPX transport directly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Always import base classes and errors (lightweight, no external deps)
from seqoria_agent.providers.base_provider import AsyncBaseProvider, BaseProvider
from seqoria_agent.providers.errors import (
    ConnectionError,
    ContentFilterError,
    ContextLengthExceededError,
    EnvironmentError,
    InvalidRequestError,
    LLMError,
    MaxRetriesExceededError,
    ModelNotFoundError,
    QuotaExceededError,
    ServerError,
)

# Type checking imports (no runtime cost)
if TYPE_CHECKING:
    from seqoria_agent.providers.openai_provider import AsyncOpenAIProvider, OpenAIProvider
    from seqoria_agent.providers.openai_compatible_provider import (
        AsyncOpenAICompatibleProvider,
        OpenAICompatibleProvider,
    )
    from seqoria_agent.providers.anthropic_provider import AsyncAnthropicProvider, AnthropicProvider
    from seqoria_agent.providers.ollama_provider import AsyncOllamaProvider, OllamaProvider


__all__ = [
    # Base classes
    "AsyncBaseProvider",
    "BaseProvider",
    # OpenAI providers (lazy loaded)
    "AsyncOpenAIProvider",
    "OpenAIProvider",
    # OpenAI-compatible HTTPX providers (lazy loaded)
    "AsyncOpenAICompatibleProvider",
    "OpenAICompatibleProvider",
    # Anthropic providers (lazy loaded)
    "AsyncAnthropicProvider",
    "AnthropicProvider",
    # Ollama providers (lazy loaded)
    "AsyncOllamaProvider",
    "OllamaProvider",
    # Errors (always available)
    "LLMError",
    "EnvironmentError",
    "ConnectionError",
    "ModelNotFoundError",
    "ContextLengthExceededError",
    "ContentFilterError",
    "MaxRetriesExceededError",
    "InvalidRequestError",
    "QuotaExceededError",
    "ServerError",
    # Utility functions
    "get_provider",
    "list_available_providers",
]


# ---------------------------------------------------------------------------
# Lazy loading implementation
# ---------------------------------------------------------------------------

# Map provider names to their module and class name.
_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    # name: (module_path, class_name)
    "AsyncOpenAIProvider": ("seqoria_agent.providers.openai_provider", "AsyncOpenAIProvider"),
    "OpenAIProvider": ("seqoria_agent.providers.openai_provider", "OpenAIProvider"),
    "AsyncOpenAICompatibleProvider": (
        "seqoria_agent.providers.openai_compatible_provider",
        "AsyncOpenAICompatibleProvider",
    ),
    "OpenAICompatibleProvider": (
        "seqoria_agent.providers.openai_compatible_provider",
        "OpenAICompatibleProvider",
    ),
    "AsyncAnthropicProvider": ("seqoria_agent.providers.anthropic_provider", "AsyncAnthropicProvider"),
    "AnthropicProvider": ("seqoria_agent.providers.anthropic_provider", "AnthropicProvider"),
    "AsyncOllamaProvider": ("seqoria_agent.providers.ollama_provider", "AsyncOllamaProvider"),
    "OllamaProvider": ("seqoria_agent.providers.ollama_provider", "OllamaProvider"),
}

# Cache for loaded providers
_loaded_providers: dict[str, type] = {}


def _check_package_available(package_name: str) -> bool:
    """Check if a package is installed without importing it."""
    try:
        importlib.util.find_spec(package_name)
        return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _load_provider(name: str) -> type:
    """Lazy load a provider class by name."""
    if name in _loaded_providers:
        return _loaded_providers[name]
    
    if name not in _PROVIDER_REGISTRY:
        raise AttributeError(f"Provider '{name}' not found in seqoria_agent.providers")
    
    module_path, class_name = _PROVIDER_REGISTRY[name]
    
    # Import the module and get the class
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    
    # Cache for future use
    _loaded_providers[name] = provider_class
    return provider_class


def __getattr__(name: str) -> Any:
    """Lazy load provider classes on first access."""
    if name in _PROVIDER_REGISTRY:
        return _load_provider(name)
    raise AttributeError(f"module 'seqoria_agent.providers' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_provider(name: str, async_variant: bool = True) -> type:
    """Get a provider class by name.
    
    Parameters
    ----------
    name : str
        Provider name (e.g., "openai", "anthropic", "ollama").
    async_variant : bool
        If True, return the async variant (default). If False, return sync.
        
    Returns
    -------
    type
        The provider class.
        
    Raises
    ------
    ValueError
        If the provider name is not recognized.
        
    Example
    -------
    >>> provider_cls = get_provider("openai")
    >>> provider = provider_cls(model_id="gpt-4o")
    """
    name_lower = name.lower()
    
    provider_map = {
        "openai": ("AsyncOpenAIProvider", "OpenAIProvider"),
        "openai_compatible": ("AsyncOpenAICompatibleProvider", "OpenAICompatibleProvider"),
        "anthropic": ("AsyncAnthropicProvider", "AnthropicProvider"),
        "ollama": ("AsyncOllamaProvider", "OllamaProvider"),
    }
    
    if name_lower not in provider_map:
        available = ", ".join(provider_map.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    
    async_name, sync_name = provider_map[name_lower]
    target_name = async_name if async_variant else sync_name
    
    return _load_provider(target_name)


def list_available_providers() -> dict[str, bool]:
    """List all providers and their availability status.
    
    Returns
    -------
    dict[str, bool]
        Mapping of provider names to whether they're available (package installed).
        
    Example
    -------
    >>> list_available_providers()
    {'openai': True, 'anthropic': True, 'ollama': False}
    """
    # All first-party providers now share HTTPX transport.
    available = _check_package_available("httpx")
    return {
        "openai": available,
        "openai_compatible": available,
        "anthropic": available,
        "ollama": available,
    }
