from __future__ import annotations

"""Centralised settings object for *AIClient* / *AsyncAIClient*.

The :class:`ClientSettings` dataclass groups the most commonly adjusted runtime
configuration values for :class:`seqoria_agent.client.AIClient` and
:class:`seqoria_agent.client.AsyncAIClient`.  Supplying a pre-constructed
settings object groups reusable defaults for application startup.

Explicit keyword arguments passed to :class:`AIClient` or :class:`AsyncAIClient`
continue to take precedence over the values held in a :class:`ClientSettings`
instance.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Union
import logging

from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.providers.base_provider import AsyncBaseProvider, BaseProvider
from seqoria_agent.client.parameters import ParameterSource, ParameterRegistry

__all__: list[str] = ["ClientSettings"]


@dataclass(slots=True)
class ClientSettings:  # noqa: D101 – succinct due to dataclass auto-doc
    # Core ------------------------------------------------------------------
    providers: List[Union[AsyncBaseProvider, BaseProvider]] = field(default_factory=list)
    mcp_servers: Iterable[str | MCPServerConfig] | None = None
    local_toolbox: LocalToolbox | list[LocalToolbox] | None = None
    logger: logging.Logger | None = None

    # Registry / executor ---------------------------------------------------
    refresh_interval: float = 0.0
    max_concurrency: int | None = None
    tool_timeout: float | None = None
    retry_policy: dict[str, Any] | None = None
    breaker_threshold: int = 5
    breaker_reset_after: int = 60
    param_registry: Mapping[str, ParameterSource] | ParameterRegistry | None = None
    allow_override: bool = False
    # Tools ----------------------------------------------------------------
    # Names of tools that must NOT be surfaced to the LLM (i.e., excluded from the
    # "tools" payload sent to providers). This does not prevent direct execution
    # via `call_mcp_tool()` if the caller passes the name explicitly.
    #
    # Stored as an immutable tuple to keep `ClientSettings.copy()` safe (shallow copy).
    blocked_tools: tuple[str, ...] = ()

    # ----------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------
    def with_provider(self, provider: Union[AsyncBaseProvider, BaseProvider]) -> "ClientSettings":
        """Return a *copy* with *provider* appended to :attr:`providers`."""
        new = self.copy()
        new.providers.append(provider)
        return new

    def copy(self) -> "ClientSettings":  # noqa: D401 – tiny helper
        """Copy configuration containers while retaining provider/resource ownership."""
        import copy

        result = copy.copy(self)
        result.providers = list(self.providers)
        if self.retry_policy is not None:
            result.retry_policy = dict(self.retry_policy)
        if isinstance(self.mcp_servers, list):
            result.mcp_servers = list(self.mcp_servers)
        if isinstance(self.local_toolbox, list):
            result.local_toolbox = list(self.local_toolbox)
        return result

    # Provide a readable string representation helpful in debug logs
    def __repr__(self) -> str:  # noqa: D401
        cls = self.__class__.__name__
        return (
            f"{cls}(mcp_servers={'configured' if self.mcp_servers is not None else 'None'}, "
            f"providers={len(self.providers)} providers,"
            f" refresh_interval={self.refresh_interval})"
        )
