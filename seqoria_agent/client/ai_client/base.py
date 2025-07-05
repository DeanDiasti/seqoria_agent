from __future__ import annotations

"""AIClient orchestrator (refactored)."""

import asyncio, logging
from typing import Any, AsyncGenerator, Iterable, List, Self, Tuple, Union

try:  # Prefer structlog but fall back to stdlib
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None  # type: ignore

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.auth.providers import BaseAuthProvider, NoAuthProvider
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.history import build_system_prompt
from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.cache import CacheProtocol, NullCache  # noqa: F401
from seqoria_agent.client.parameters import ParameterProvider
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.settings import ClientSettings  # noqa: F401
from seqoria_agent.models.chunk import AssistantMessage, SystemMessage, ToolCallRequest, ToolCallResult, ChatChunk, UserMessage, ToolCallRequestList, LLMUsage, ToolMessages

__all__ = ["AIClient"]

class AIClient:  # noqa: D101 – tiny glue layer
    def __init__(self,
                 providers: List[BaseProvider] | None = None,
                 *,
                 urls: Iterable[str] | None = None,
                 local_toolbox: LocalToolbox | list[LocalToolbox] | None = None,
                 logger: logging.Logger | None = None,
                 refresh_interval: float = 0,
                 cache: CacheProtocol | None = None,
                 cache_ttl: int | None = 300,
                 max_concurrency: int | None = None,
                 tool_timeout: float | None = None,
                 retry_policy: dict[str, Any] | None = None,
                 breaker_threshold: int = 5,
                 breaker_reset_after: int = 60,
                 param_providers: list[ParameterProvider] | None = None,
                 allow_override: bool = False,
                 enable_meta_tools: bool | Iterable[str] = True,
                 settings: ClientSettings | None = None,
                 auth_provider: BaseAuthProvider | None = None,
                 system_prompt: str | None = None,
                 **client_kwargs: Any) -> None:
        if settings is None:
            settings = ClientSettings()

        # Use settings values only when the corresponding explicit argument is not supplied
        providers = providers or settings.providers
        urls = urls if urls is not None else settings.urls
        local_toolbox = local_toolbox if local_toolbox is not None else settings.local_toolbox
        logger = logger if logger is not None else settings.logger
        refresh_interval = refresh_interval if refresh_interval != 0 else settings.refresh_interval
        cache = cache if cache is not None else settings.cache
        cache_ttl = cache_ttl if cache_ttl is not None else settings.cache_ttl
        max_concurrency = max_concurrency if max_concurrency is not None else settings.max_concurrency
        tool_timeout = tool_timeout if tool_timeout is not None else settings.tool_timeout
        retry_policy = retry_policy if retry_policy is not None else settings.retry_policy
        breaker_threshold = breaker_threshold if breaker_threshold != 5 else settings.breaker_threshold
        breaker_reset_after = (breaker_reset_after if breaker_reset_after != 60 else settings.breaker_reset_after)
        param_providers = param_providers if param_providers is not None else settings.param_providers

        self._auth_provider: BaseAuthProvider = auth_provider or NoAuthProvider()
        if "auth_provider" not in client_kwargs:
            client_kwargs["auth_provider"] = self._auth_provider
        final_client_kwargs = {**settings.client_kwargs, **client_kwargs}

        if not providers:
            raise ValueError("'providers' may not be empty")
        self._providers = providers
        self._urls = list(dict.fromkeys(urls or []))
        self._boxes = [] if local_toolbox is None else (local_toolbox if isinstance(local_toolbox, list) else [local_toolbox])
        # Always use stdlib logging for internal library logs to avoid structlog API incompatibilities
        self._logger = logger or logging.getLogger(name="seqoria_agent")
        self._tool_map: dict[str, Any] = {}

        self.pool = ConnectionPool(urls=self._urls,
                                   logger=self._logger,
                                   retry_policy=retry_policy or {},
                                   breaker_threshold=breaker_threshold,
                                   breaker_reset_after=breaker_reset_after,
                                   tool_map=self._tool_map,
                                   client_kwargs=final_client_kwargs)
        self.registry = ToolRegistry(pool=self.pool,
                                     local_toolboxes=self._boxes,
                                     tool_map=self._tool_map,
                                     refresh_interval=refresh_interval,
                                     logger=self._logger,
                                     enable_meta_tools=enable_meta_tools)
        self.executor = Executor(pool=self.pool,
                                 registry=self.registry,
                                 cache=cache or NullCache(),
                                 cache_ttl=cache_ttl,
                                 tool_semaphore=asyncio.Semaphore(value=max_concurrency) if max_concurrency else None,
                                 tool_timeout=tool_timeout,
                                 param_providers=param_providers,
                                 allow_override=allow_override,
                                 logger=self._logger)
        self.llm = LLMInterface(providers=self._providers, logger=self._logger)
        # System prompt handling ------------------------------------------------
        if system_prompt == "":  # Explicitly disable automatic prompt
            self._system_prompt: str | None = None
        elif system_prompt is None:
            self._system_prompt = build_system_prompt()
        else:
            self._system_prompt = system_prompt.strip()

    # ------------------------------------------------------------------ context
    async def __aenter__(self) -> Self:
        await self.pool.start(); await self.registry.refresh()
        if not self.pool.has_servers and not self._boxes:
            self._logger.warning(msg="AIClient initialised without tools - LLM-only mode")
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        await self.pool.close(); await self.registry.close(); self._tool_map.clear()

    # ------------------------------------------------------------------ util & compat
    async def call_mcp_tool(self, name: str, arguments: dict[str, Any], *, server_args: dict | None = None):
        return await self.pool.call_tool(name=name, arguments=arguments, server_args=server_args)

    async def refresh_tools(self) -> None:  # manual refresh
        await self.registry.refresh()

    async def run_tool_calls(self, tool_calls: List[ToolCallRequest]) -> List[ToolCallResult]:
        return await self.executor.run_calls(tool_calls=tool_calls)
    
    def run_call_stream(self, tool_calls: List[ToolCallRequest]) -> AsyncGenerator[Tuple[ToolCallRequest, ToolCallResult], None]:
        return self.executor.run_call_stream(tool_calls=tool_calls)

    # ------------------------------------------------------------------ helpers (public)

    def tools_enabled(self) -> bool:  # noqa: D401
        """Return *True* if any tool is available."""

        return bool(self._tool_map or self._boxes)

    # ------------------------------------------------------------------ LLM single-shot helpers

    async def llm_generate_once(self,
                                history: List[dict],
                                *,
                                use_tools: bool = True,
                                tool_choice: str = "auto",
                                model: str | None = None) -> Union[AssistantMessage, ToolCallRequestList]:
        """Single LLM step (optionally with tools)."""

        if history is None:
            raise ValueError("history required")

        # Ensure system prompt present
        self._ensure_system_prompt(history=history)

        if use_tools and self.tools_enabled():
            resp = await self.llm.generate(history=history,
                                           tools=self.registry.llm_tools,
                                           tool_choice=tool_choice,
                                           model=model)
        else:
            resp = await self.llm.generate(history=history, 
                                           model=model)

        return resp

    async def llm_stream_once(self,
                              history: List[dict],
                              *,
                              use_tools: bool = True,
                              tool_choice: str = "auto",
                              model: str | None = None) -> AsyncGenerator[Union[str, ChatChunk, AssistantMessage, ToolCallRequestList], None]:
        """Stream a single LLM step"""

        if history is None:
            raise ValueError("history required")
        
        # Ensure system prompt present
        self._ensure_system_prompt(history=history)

        if use_tools and self.tools_enabled():
            async for chunk in self.llm.stream(history=history, 
                                               tools=self.registry.llm_tools, 
                                               tool_choice=tool_choice,
                                               model=model):
                yield chunk
        else:
            async for chunk in self.llm.stream(history=history, 
                                               model=model):
                yield chunk

    async def chat(self, 
                   user_query: str, 
                   history: List[dict], 
                   *, 
                   max_turns: int = 5,
                   model: str | None = None) -> AsyncGenerator[Union[AssistantMessage, ToolCallRequestList, ToolMessages], None]:  # noqa: D401
        """Chat with the LLM."""
        if history is None:
            raise ValueError("history required")

        try:
            # Inject system prompt before any new messages
            self._ensure_system_prompt(history=history)
            
            history.append(UserMessage(role="user", content=user_query))
            
            # LLM-only fast path --------------------------------------------------
            if not self.tools_enabled():
                yield await self.llm_generate_once(history=history, 
                                                    use_tools=False,
                                                    model=model)
                return
            

            total_usage = LLMUsage(prompt_tokens=0, 
                                   completion_tokens=0, 
                                   total_tokens=0)

            def _add_usage(u) -> None:  # noqa: D401
                if u is None:
                    return
                total_usage.prompt_tokens += u.prompt_tokens or 0
                total_usage.completion_tokens += u.completion_tokens or 0
                total_usage.total_tokens += u.total_tokens or 0


            for _ in range(max_turns):
                resp = await self.llm_generate_once(history=history, 
                                                    use_tools=True, 
                                                    tool_choice="auto",
                                                    model=model)
                
                _add_usage(u=resp.usage)

                if isinstance(resp, AssistantMessage):
                    yield resp
                    return

                if isinstance(resp, ToolCallRequestList):
                    # Assistant decided to use tool(s) – no textual answer yet
                    tool_results = await self.executor.run_calls(tool_calls=resp.tool_calls)

                    # Interleave requests and results so each tool_call is immediately
                    # followed by its corresponding tool_result.
                    tool_messages = ToolMessages()
                    for req, res in zip(resp.tool_calls, tool_results):
                        tool_messages.extend([req, res])

                    history.extend(tool_messages)
                    yield tool_messages


            else:
                # Turn limit reached – return a structured response so callers can resume later
                yield AssistantMessage(role="assistant", 
                                       content="[Turn limit reached]", 
                                       model=None, 
                                       usage=total_usage)
        except Exception as e:
            raise e


    # ------------------------------------------------------------------ streaming chat

    async def stream_chat(self, 
                          user_query: str, 
                          history: List[dict], 
                          *, 
                          max_turns: int = 5, 
                          model: str | None = None) -> AsyncGenerator[Union[str, ChatChunk, AssistantMessage, ToolMessages], None]:  # noqa: D401
        """Stream a chat session."""
        if history is None:
            raise ValueError("history required")
        if user_query is None:
            raise ValueError("user_query required")
        
        # Ensure system prompt present
        self._ensure_system_prompt(history=history)

        history.append(UserMessage(role="user", 
                                   content=user_query))

        try:
            # LLM-only fast path --------------------------------------------------
            if not self.tools_enabled():
                async for chunk in self.llm_stream_once(history=history, 
                                                        use_tools=False,  
                                                        model=model):
                    yield chunk

                return


            total_usage = LLMUsage(prompt_tokens=0, 
                                   completion_tokens=0, 
                                   total_tokens=0)

            def _add_usage(u) -> None:  # noqa: D401
                if u is None:
                    return
                total_usage.prompt_tokens += u.prompt_tokens or 0
                total_usage.completion_tokens += u.completion_tokens or 0
                total_usage.total_tokens += u.total_tokens or 0


            for _ in range(max_turns):
                async for c in self.llm_stream_once(history=history, 
                                                    use_tools=True, 
                                                    tool_choice="auto", 
                                                    model=model):
                    
                    if isinstance(c, ChatChunk):
                        yield c

                    else:
                        _add_usage(u=c.usage)


                        if isinstance(c, AssistantMessage):
                            message = AssistantMessage(role="assistant", 
                                                       content=c.content, 
                                                       model=c.model, 
                                                       usage=total_usage)
                            history.append(message)
                            yield message

                            return

                        elif isinstance(c, ToolCallRequestList):
                            tool_results = await self.executor.run_calls(tool_calls=c.tool_calls)

                            # Interleave requests and results so each tool_call is immediately
                            # followed by its corresponding tool_result.
                            tool_messages = ToolMessages()
                            for req, res in zip(c.tool_calls, tool_results):
                                tool_messages.extend([req, res])

                            history.extend(tool_messages)
                            yield tool_messages

                            break

                        else:
                            raise ValueError("Unknown chunk type from provider stream: %s", type(c))

            else:
                # Turn limit reached --------------------------------------------------
                yield AssistantMessage(role="assistant", 
                                       content="[Turn limit reached]", 
                                       model=None, 
                                       usage=total_usage)

        except Exception as e:
            raise e

    # ------------------------------------------------------------------ auth

    def add_auth_provider(self, provider: BaseAuthProvider) -> None:  # noqa: D401
        """Replace the active *provider* for outgoing HTTP requests.

        The change is applied immediately to existing transport connections
        so subsequent calls use the updated credentials.
        """
        self._auth_provider = provider

        # Update all active transports so they pick up the new provider.
        for conn in self.pool.connections:
            if hasattr(conn.client, "set_auth_provider"):
                try:
                    conn.client.set_auth_provider(provider=provider)
                except Exception:
                    continue

    # ------------------------------------------------------------------ internal helpers

    def _ensure_system_prompt(self, history: List[dict]) -> None:  # noqa: D401
        """Insert the system prompt at the beginning of *history* once.

        The helper is idempotent - if the first entry is already a system
        message, it does nothing.  Automatic injection can be disabled by
        passing an empty string ("") for *system_prompt* in the constructor.
        """
        if self._system_prompt is None:
            return  # disabled

        if not history or history[0].get("role") != "system":
            history.insert(0, SystemMessage(role="system", 
                                            content=self._system_prompt))
