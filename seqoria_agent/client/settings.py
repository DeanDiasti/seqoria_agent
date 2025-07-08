from __future__ import annotations

"""Centralised settings object for *AIClient*.

The :class:`ClientSettings` dataclass groups the most commonly adjusted runtime
configuration values for :class:`seqoria_agent.client.AIClient`.  Supplying a
pre-constructed settings object keeps application code tidy when many knobs
need tweaking while **not** breaking the existing constructor signature.

Explicit keyword arguments passed to :class:`AIClient` continue to take
precedence over the values held in a :class:`ClientSettings` instance.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping
import logging

from seqoria_agent.client.cache import CacheProtocol  # noqa: F401
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.client.parameters import ParameterSource, ParameterRegistry

__all__: list[str] = ["ClientSettings"]


@dataclass(slots=True)
class ClientSettings:  # noqa: D101 – succinct due to dataclass auto-doc
    # Core ------------------------------------------------------------------
    providers: List[BaseProvider] = field(default_factory=list)
    urls: Iterable[str] | None = None
    local_toolbox: LocalToolbox | list[LocalToolbox] | None = None
    logger: logging.Logger | None = None

    # Registry / executor ---------------------------------------------------
    refresh_interval: float = 0.0
    cache: CacheProtocol | None = None  # When None AIClient will use NullCache
    cache_ttl: int | None = 300
    max_concurrency: int | None = None
    tool_timeout: float | None = None
    retry_policy: dict[str, Any] | None = None
    breaker_threshold: int = 5
    breaker_reset_after: int = 60
    param_registry: Mapping[str, ParameterSource] | ParameterRegistry | None = None
    allow_override: bool = False

    # Transport customisation ----------------------------------------------
    client_kwargs: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------
    def with_provider(self, provider: BaseProvider) -> "ClientSettings":
        """Return a *copy* with *provider* appended to :attr:`providers`."""
        new = self.copy()
        new.providers.append(provider)
        return new

    # Simple shallow copy since we only store primitives / references         
    def copy(self) -> "ClientSettings":  # noqa: D401 – tiny helper
        import copy

        return copy.copy(self)

    # Provide a readable string representation helpful in debug logs
    def __repr__(self) -> str:  # noqa: D401
        cls = self.__class__.__name__
        return (
            f"{cls}(urls={self.urls}, providers={len(self.providers)} providers,"  # avoid verbose repr
            f" cache={self.cache or 'NullCache'}, refresh_interval={self.refresh_interval})"
        ) 