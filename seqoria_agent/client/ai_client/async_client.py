"""AsyncAIClient - MCP-native AI orchestrator (async version).

This module provides the AsyncAIClient class for building AI applications
with tool support via the Model Context Protocol (MCP).

Features:
- Multi-provider LLM support (OpenAI, Anthropic, etc.)
- MCP server connections (HTTP, SSE, stdio)
- Local toolbox support
- Circuit breaker and retry resilience
- Caching for tool results
- Server-populated parameter injection

For synchronous usage, see :class:`AIClient` in ``sync_client.py``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Iterable, List, Mapping, Self, Tuple, Union

from seqoria_agent.auth.providers import BaseAuthProvider, NoAuthProvider
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.history import build_system_prompt
from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.cache import CacheProtocol, NullCache
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.client.parameters import ParamContext, ParameterRegistry, ParameterSource
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.models.chunk import (
    AssistantMessage,
    ChatChunk,
    LLMUsage,
    ToolCallRequest,
    ToolCallRequestList,
    ToolCallResult,
    ToolMessages,
    UserMessage,
)
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.toolbox import LocalToolbox

__all__ = ["AsyncAIClient"]


class AsyncAIClient:
    """MCP-native AI client for tool-augmented LLM interactions (async version).

    This is the asynchronous version of the AI client. For synchronous usage,
    use :class:`AIClient` instead.

    Parameters
    ----------
    providers : list[AsyncBaseProvider]
        LLM providers to use (required).
    mcp_servers : Iterable[str | MCPServerConfig] | None
        MCP servers to connect to (URLs or configs).
    local_toolbox : LocalToolbox | list[LocalToolbox] | None
        Local toolboxes for in-process tool execution.
    logger : logging.Logger | None
        Logger for client events.
    refresh_interval : float
        Interval for automatic tool registry refresh (0 = disabled).
    cache : CacheProtocol | None
        Cache for tool results.
    cache_ttl : int | None
        Cache TTL in seconds.
    max_concurrency : int | None
        Maximum concurrent tool executions.
    tool_timeout : float | None
        Timeout for tool execution in seconds.
    retry_policy : dict | None
        Retry configuration with "max_attempts" and "backoff_base".
    breaker_threshold : int
        Failures before circuit trips.
    breaker_reset_after : int
        Seconds before circuit resets.
    param_registry : Mapping | ParameterRegistry | None
        Server-populated parameter sources.
    allow_override : bool
        Allow LLM to override server-populated params.
    enable_meta_tools : bool | Iterable[str]
        Enable built-in meta-tools.
    settings : ClientSettings | None
        Default settings (overridden by explicit args).
    auth_provider : BaseAuthProvider | None
        Authentication provider for MCP requests.
    system_prompt : str | None
        System prompt (None = default, "" = disabled).

    Examples
    --------
    >>> async with AsyncAIClient(providers=[AsyncOpenAIProvider()]) as client:
    ...     async for chunk in client.stream_chat("Hello!", history=[]):
    ...         print(chunk)
    """

    def __init__(
        self,
        providers: List[AsyncBaseProvider] | None = None,
        *,
        # MCP server connections
        mcp_servers: Iterable[str | MCPServerConfig] | None = None,
        # Legacy alias for mcp_servers
        urls: Iterable[str] | None = None,
        # Local tools
        local_toolbox: LocalToolbox | list[LocalToolbox] | None = None,
        # Logging
        logger: logging.Logger | None = None,
        # Tool registry
        refresh_interval: float = 0,
        # Caching
        cache: CacheProtocol | None = None,
        cache_ttl: int | None = 300,
        # Execution
        max_concurrency: int | None = None,
        tool_timeout: float | None = None,
        # Resilience
        retry_policy: dict[str, Any] | None = None,
        breaker_threshold: int = 5,
        breaker_reset_after: int = 60,
        # Parameters
        param_registry: Union[
            Mapping[str, ParameterSource], ParameterRegistry, None
        ] = None,
        allow_override: bool = False,
        # Meta-tools
        enable_meta_tools: bool | Iterable[str] = True,
        # Settings
        settings: ClientSettings | None = None,
        # Auth
        auth_provider: BaseAuthProvider | None = None,
        # System prompt
        system_prompt: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        if settings is None:
            settings = ClientSettings()

        # Resolve settings with explicit overrides
        providers = providers or settings.providers
        
        # Merge mcp_servers and legacy urls
        _servers: list[str | MCPServerConfig] = []
        if mcp_servers is not None:
            _servers.extend(mcp_servers)
        elif urls is not None:
            _servers.extend(urls)
        elif settings.urls:
            _servers.extend(settings.urls)

        local_toolbox = (
            local_toolbox if local_toolbox is not None else settings.local_toolbox
        )
        logger = logger if logger is not None else settings.logger
        refresh_interval = (
            refresh_interval if refresh_interval != 0 else settings.refresh_interval
        )
        cache = cache if cache is not None else settings.cache
        cache_ttl = cache_ttl if cache_ttl is not None else settings.cache_ttl
        max_concurrency = (
            max_concurrency if max_concurrency is not None else settings.max_concurrency
        )
        tool_timeout = (
            tool_timeout if tool_timeout is not None else settings.tool_timeout
        )
        retry_policy = (
            retry_policy if retry_policy is not None else settings.retry_policy
        )
        breaker_threshold = (
            breaker_threshold if breaker_threshold != 5 else settings.breaker_threshold
        )
        breaker_reset_after = (
            breaker_reset_after
            if breaker_reset_after != 60
            else settings.breaker_reset_after
        )
        param_registry = (
            param_registry
            if param_registry is not None
            else getattr(settings, "param_registry", None)
        )

        # Validate
        if not providers:
            raise ValueError("'providers' may not be empty")

        # Store configuration
        self._providers = providers
        self._servers = _servers
        self._boxes = (
            []
            if local_toolbox is None
            else (
                local_toolbox
                if isinstance(local_toolbox, list)
                else [local_toolbox]
            )
        )
        self._logger = logger or logging.getLogger(name="seqoria_agent")
        self._auth_provider: BaseAuthProvider = auth_provider or NoAuthProvider()
        self._tool_map: dict[str, Any] = {}

        # Build MCP server configs
        server_configs = self._build_server_configs(_servers, client_kwargs)

        # Initialize components
        self.pool = MCPConnectionPool(
            servers=server_configs,
            logger=self._logger,
            retry_policy=retry_policy or {},
            breaker_threshold=breaker_threshold,
            breaker_reset_after=breaker_reset_after,
            tool_map=self._tool_map,
        )

        self.registry = ToolRegistry(
            pool=self.pool,
            local_toolboxes=self._boxes,
            tool_map=self._tool_map,
            refresh_interval=refresh_interval,
            logger=self._logger,
            enable_meta_tools=enable_meta_tools,
        )

        self.executor = Executor(
            pool=self.pool,
            registry=self.registry,
            cache=cache or NullCache(),
            cache_ttl=cache_ttl,
            tool_semaphore=(
                asyncio.Semaphore(value=max_concurrency) if max_concurrency else None
            ),
            tool_timeout=tool_timeout,
            param_registry=param_registry,
            allow_override=allow_override,
            logger=self._logger,
        )

        self.llm = LLMInterface(providers=self._providers, logger=self._logger)

        # System prompt handling
        if system_prompt == "":
            self._system_prompt: str | None = None
        elif system_prompt is None:
            self._system_prompt = build_system_prompt()
        else:
            self._system_prompt = system_prompt.strip()

    def _build_server_configs(
        self,
        servers: list[str | MCPServerConfig],
        client_kwargs: dict[str, Any],
    ) -> list[MCPServerConfig]:
        """Convert server URLs to MCPServerConfig objects."""
        configs = []

        for server in servers:
            if isinstance(server, MCPServerConfig):
                configs.append(server)
            else:
                configs.append(MCPServerConfig(url=server))

        return configs

    # Context manager methods
    async def __aenter__(self) -> Self:
        await self.pool.start()
        await self.registry.refresh()
        if not self.pool.has_servers and not self._boxes:
            self._logger.warning("AsyncAIClient initialized without tools - LLM-only mode")
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        await self.pool.close()
        await self.registry.close()
        self._tool_map.clear()

    # Tool execution methods
    async def call_mcp_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        server_args: dict[str, Any] | None = None,
    ) -> List[Any]:
        """Call an MCP tool directly."""
        return await self.pool.call_tool(
            name=name, arguments=arguments, server_args=server_args
        )

    async def refresh_tools(self) -> None:
        """Manually refresh the tool registry."""
        await self.registry.refresh()

    async def run_tool_calls(
        self,
        tool_calls: List[ToolCallRequest],
        *,
        context: ParamContext | None = None,
    ) -> List[ToolCallResult]:
        """Execute tool calls and return results."""
        return await self.executor.run_calls(
            tool_calls=tool_calls, context=context or {}
        )

    def run_call_stream(
        self,
        tool_calls: List[ToolCallRequest],
        *,
        context: ParamContext | None = None,
    ) -> AsyncGenerator[Tuple[ToolCallRequest, ToolCallResult], None]:
        """Stream tool call results as they complete."""
        return self.executor.run_call_stream(
            tool_calls=tool_calls, context=context or {}
        )

    def tools_enabled(self) -> bool:
        """Return True if any tools are available."""
        return bool(self._tool_map or self._boxes)

    # LLM interaction methods
    async def llm_generate_once(
        self,
        history: List[dict],
        *,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> Union[AssistantMessage, ToolCallRequestList]:
        """Single LLM generation step."""
        if history is None:
            raise ValueError("history required")

        self._ensure_system_prompt(history=history)

        if use_tools and self.tools_enabled():
            return await self.llm.generate(
                history=history,
                tools=self.registry._llm_tools_view(),
                tool_choice=tool_choice,
                model=model,
            )
        else:
            return await self.llm.generate(history=history, model=model)

    async def llm_stream_once(
        self,
        history: List[dict],
        *,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> AsyncGenerator[
        Union[str, ChatChunk, AssistantMessage, ToolCallRequestList], None
    ]:
        """Stream a single LLM generation step."""
        if history is None:
            raise ValueError("history required")

        self._ensure_system_prompt(history=history)

        if use_tools and self.tools_enabled():
            async for chunk in self.llm.stream(
                history=history,
                tools=self.registry._llm_tools_view(),
                tool_choice=tool_choice,
                model=model,
            ):
                yield chunk
        else:
            async for chunk in self.llm.stream(history=history, model=model):
                yield chunk

    async def chat(
        self,
        user_query: str,
        history: List[dict],
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
    ) -> AsyncGenerator[
        Union[AssistantMessage, ToolCallRequestList, ToolMessages], None
    ]:
        """Chat with the LLM, handling tool calls automatically."""
        if history is None:
            raise ValueError("history required")

        tool_ctx: ParamContext = context or {}

        self._ensure_system_prompt(history=history)
        history.append(UserMessage(role="user", content=user_query))

        # LLM-only fast path
        if not self.tools_enabled():
            yield await self.llm_generate_once(
                history=history, use_tools=False, model=model
            )
            return

        total_usage = LLMUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

        def _add_usage(u: Any) -> None:
            if u is None:
                return
            total_usage.prompt_tokens += u.prompt_tokens or 0
            total_usage.completion_tokens += u.completion_tokens or 0
            total_usage.total_tokens += u.total_tokens or 0

        for _ in range(max_turns):
            resp = await self.llm_generate_once(
                history=history,
                use_tools=True,
                tool_choice="auto",
                model=model,
            )

            _add_usage(resp.usage)

            if isinstance(resp, AssistantMessage):
                yield resp
                return

            if isinstance(resp, ToolCallRequestList):
                tool_results = await self.executor.run_calls(
                    tool_calls=resp.tool_calls, context=tool_ctx
                )

                tool_messages = ToolMessages()
                for req, res in zip(resp.tool_calls, tool_results):
                    tool_messages.extend([req, res])

                history.extend(tool_messages)
                yield tool_messages

        # Turn limit reached
        yield AssistantMessage(
            role="assistant",
            content="[Turn limit reached]",
            model=None,
            usage=total_usage,
        )

    async def stream_chat(
        self,
        user_query: str,
        history: List[dict],
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
    ) -> AsyncGenerator[
        Union[str, ChatChunk, AssistantMessage, ToolMessages], None
    ]:
        """Stream a chat session with the LLM."""
        if history is None:
            raise ValueError("history required")
        if user_query is None:
            raise ValueError("user_query required")

        tool_ctx: ParamContext = context or {}

        self._ensure_system_prompt(history=history)
        history.append(UserMessage(role="user", content=user_query))

        # LLM-only fast path
        if not self.tools_enabled():
            async for chunk in self.llm_stream_once(
                history=history, use_tools=False, model=model
            ):
                yield chunk
            return

        total_usage = LLMUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

        def _add_usage(u: Any) -> None:
            if u is None:
                return
            total_usage.prompt_tokens += u.prompt_tokens or 0
            total_usage.completion_tokens += u.completion_tokens or 0
            total_usage.total_tokens += u.total_tokens or 0

        for _ in range(max_turns):
            async for c in self.llm_stream_once(
                history=history,
                use_tools=True,
                tool_choice="auto",
                model=model,
            ):
                if isinstance(c, ChatChunk):
                    yield c
                else:
                    _add_usage(c.usage)

                    if isinstance(c, AssistantMessage):
                        message = AssistantMessage(
                            role="assistant",
                            content=c.content,
                            model=c.model,
                            usage=total_usage,
                        )
                        history.append(message)
                        yield message
                        return

                    elif isinstance(c, ToolCallRequestList):
                        tool_results = await self.executor.run_calls(
                            tool_calls=c.tool_calls, context=tool_ctx
                        )

                        tool_messages = ToolMessages()
                        for req, res in zip(c.tool_calls, tool_results):
                            tool_messages.extend([req, res])

                        history.extend(tool_messages)
                        yield tool_messages
                        break

                    else:
                        raise ValueError(f"Unknown chunk type: {type(c)}")

        # Turn limit reached
        yield AssistantMessage(
            role="assistant",
            content="[Turn limit reached]",
            model=None,
            usage=total_usage,
        )

    def add_auth_provider(self, provider: BaseAuthProvider) -> None:
        """Replace the authentication provider for MCP requests."""
        self._auth_provider = provider

    def _ensure_system_prompt(self, history: List[dict]) -> None:
        """Insert system prompt at start of history if not present."""
        if self._system_prompt is None:
            return

        if not history:
            history.insert(0, {"role": "system", "content": self._system_prompt})
            return

        first = history[0]
        if isinstance(first, dict):
            if first.get("role") != "system":
                history.insert(0, {"role": "system", "content": self._system_prompt})
        else:
            if getattr(first, "role", None) != "system":
                history.insert(0, {"role": "system", "content": self._system_prompt})

