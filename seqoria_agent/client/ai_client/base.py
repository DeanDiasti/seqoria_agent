from __future__ import annotations

"""AIClient orchestrator (refactored)."""

import asyncio, logging
from typing import Any, AsyncGenerator, Iterable, List, Optional, Self, Union

try:  # Prefer structlog but fall back to stdlib
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None  # type: ignore

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.auth.providers import BaseAuthProvider, NoAuthProvider  # Phase-2
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.history import build_system_prompt
from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.cache import CacheProtocol, NullCache  # noqa: F401
from seqoria_agent.client.parameters import ParameterProvider
from seqoria_agent.models.chunk import ChatChunk
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall, LLMUsage
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.settings import ClientSettings  # noqa: F401

__all__ = ["AIClient"]


def _as_list(x) -> list:
    return [] if x is None else (x if isinstance(x, list) else [x])


class AIClient:  # noqa: D101 – tiny glue layer
    def __init__(
        self,
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
        **client_kwargs: Any,
    ) -> None:
        # ------------------------------------------------------------------
        # Phase 9 – merge explicit kwargs with ClientSettings (non-breaking)
        # ------------------------------------------------------------------
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
        breaker_reset_after = (
            breaker_reset_after if breaker_reset_after != 60 else settings.breaker_reset_after
        )
        param_providers = param_providers if param_providers is not None else settings.param_providers
        # ------------------------------------------------------------------
        # Phase-2 – authentication provider wiring
        # ------------------------------------------------------------------

        self._auth_provider: BaseAuthProvider = auth_provider or NoAuthProvider()

        # Merge auth_provider into client kwargs so ConnectionPool/Transport
        # can thread it through to every HTTP request.  Explicit overrides in
        # *client_kwargs* remain honoured.
        if "auth_provider" not in client_kwargs:
            client_kwargs["auth_provider"] = self._auth_provider

        # 'allow_override' stays as explicitly provided (default False)
        # Merge client_kwargs with settings.client_kwargs (explicit has priority)
        final_client_kwargs = {**settings.client_kwargs, **client_kwargs}

        # ------------------------------------------------------------------
        # Preserve original behaviour after merging
        # ------------------------------------------------------------------
        if not providers:
            raise ValueError("'providers' may not be empty")
        self._providers = providers
        self._urls = list(dict.fromkeys(urls or []))
        self._boxes = _as_list(local_toolbox)
        self._logger = logger or (
            structlog.get_logger("seqoria_agent")  # type: ignore
            if structlog else logging.getLogger(name="seqoria_agent")
        )
        self._tool_map: dict[str, Any] = {}

        self.pool = ConnectionPool(
            urls=self._urls,
            logger=self._logger,
            retry_policy=retry_policy or {},
            breaker_threshold=breaker_threshold,
            breaker_reset_after=breaker_reset_after,
            tool_map=self._tool_map,
            client_kwargs=final_client_kwargs,
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
            tool_semaphore=asyncio.Semaphore(value=max_concurrency) if max_concurrency else None,
            tool_timeout=tool_timeout,
            param_providers=param_providers,
            allow_override=allow_override,
            logger=self._logger,
        )
        self.llm = LLMInterface(providers=self._providers, logger=self._logger)
        self._system_prompt = build_system_prompt()
        self._hist: Optional[List[dict]] = None

    # ------------------------------------------------------------------ context
    async def __aenter__(self) -> Self:
        await self.pool.start(); await self.registry.refresh()
        if not self.pool.has_servers and not self._boxes:
            self._logger.warning("AIClient initialised without tools - LLM-only mode")
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        await self.pool.close(); await self.registry.close(); self._tool_map.clear()

    # ------------------------------------------------------------------ util & compat
    async def call_mcp_tool(self, name: str, arguments: dict[str, Any], *, server_args: dict | None = None):
        return await self.pool.call_tool(name=name, arguments=arguments, server_args=server_args)

    async def _update_llm_tools(self) -> None:
        """Legacy compatibility shim – delegates to ToolRegistry.refresh."""
        await self.registry.refresh()

    async def refresh_tools(self) -> None:  # manual refresh
        await self._update_llm_tools()

    # ------------------------------------------------------------------ Phase-1 helpers (public)

    def tools_enabled(self) -> bool:  # noqa: D401
        """Return *True* if any tool is available."""

        # Prefer the registry as source-of-truth (includes meta-tools) but
        # fall back to the historical check for edge-cases where the registry
        # has not been refreshed yet (e.g. before ``AIClient.__aenter__``).
        return bool(self._tool_map or self._boxes)

    # Deprecated private alias kept for backward-compatibility -------------
    def _tools_enabled(self) -> bool:  # noqa: D401
        """Deprecated - call :pyfunc:`tools_enabled`."""

        return self.tools_enabled()

    def prepare_history(self, user_query: str, history: List[dict]) -> List[dict]:  # noqa: D401
        """Add system prompt (if absent) and user query."""

        if history is None:
            raise ValueError("history required")

        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": self._system_prompt})

        history.append({"role": "user", "content": user_query})
        return history

    # ------------------------------------------------------------------ Phase-2 LLM single-shot helpers

    async def llm_generate_once(  # noqa: D401
        self,
        history: List[dict],
        *,
        use_tools: bool = True,
        tool_choice: str = "auto",
        include_usage: bool = False,
    ) -> Union[LLMResponse, dict]:
        """Single LLM step (optionally with tools).

        When *include_usage* is *True* the return value is a dict
        ``{"response": <LLMResponse>, "usage": <usage_dict_or_None>}``
        so callers can access token statistics without having to unzip the
        :class:`LLMResponse` object themselves.  When *False* (default) the
        behaviour is identical to previous versions and the plain
        :class:`LLMResponse` is returned for full backwards-compatibility."""

        if history is None:
            raise ValueError("history required")

        if use_tools and self.tools_enabled():
            resp = await self.llm.generate(history=history,
                                           tools=self.registry.llm_tools,
                                           tool_choice=tool_choice)
            if include_usage:
                return {"response": resp, 
                        "usage": resp.usage}
            return resp

        resp = await self.llm.generate(history=history)
        if include_usage:
            return {"response": resp, 
                    "usage": resp.usage}
        return resp

    async def llm_stream_once(  # noqa: D401
        self,
        history: List[dict],
        *,
        use_tools: bool = True,
        tool_choice: str = "auto",
        include_usage: bool = False,
    ) -> AsyncGenerator[Union[str, dict, LLMResponse], None]:
        """Stream a single LLM step.

        When *include_usage* is *True* the final :class:`LLMResponse` chunk
        is wrapped in ``{"response": <LLMResponse>, "usage": <usage_dict_or_None>}``.
        Intermediate string deltas stay untouched so existing token-by-token
        rendering code continues to work."""

        if history is None:
            raise ValueError("history required")

        async def _gen() -> AsyncGenerator[dict[str, Any] | LLMResponse | str, None]:
            if use_tools and self.tools_enabled():
                stream_iter = self.llm.stream(history=history, 
                                              tools=self.registry.llm_tools, 
                                              tool_choice=tool_choice)
            else:
                stream_iter = self.llm.stream(history=history)

            async for chunk in stream_iter:
                if isinstance(chunk, LLMResponse):
                    if include_usage:
                        yield {"response": chunk, 
                               "usage": chunk.usage}
                    else:
                        yield chunk
                else:
                    # str deltas pass through unchanged
                    yield chunk

        async for c in _gen():
            yield c

    # ------------------------------------------------------------------ Phase-3 tool execution wrapper

    async def run_tool_calls(self, tool_calls: List[ToolCall], history: List[dict]):  # noqa: D401
        """Forward to :pymeth:`Executor.run_calls`."""
        return await self.executor.run_calls(tool_calls=tool_calls, history=history)

    # ------------------------------------------------------------------ chat (refactored Phase-4)

    async def chat(self, 
                   user_query: str, 
                   history: List[dict], 
                   *, 
                   max_turns: int = 5, 
                   include_usage: bool = False) -> dict[str, Any]:  # noqa: D401
        if history is None:
            raise ValueError("history required")
        if self._hist is not None:
            raise RuntimeError("chat re-entered")
        self._hist = history
        try:
            total_usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

            def _add_usage(u) -> None:  # noqa: D401
                if u is None:
                    return
                total_usage.prompt_tokens += u.prompt_tokens or 0
                total_usage.completion_tokens += u.completion_tokens or 0
                total_usage.total_tokens += u.total_tokens or 0

            self.prepare_history(user_query=user_query, 
                                 history=history)  # Phase-1 helper
            # LLM-only fast path --------------------------------------------------
            if not self.tools_enabled():
                resp_raw = await self.llm_generate_once(history=history, 
                                                        use_tools=False, 
                                                        include_usage=include_usage)

                if include_usage:
                    resp_obj = resp_raw["response"]  # type: ignore[index]
                    usage_block = resp_raw["usage"]  # type: ignore[index]
                else:
                    resp_obj = resp_raw  # type: ignore[assignment]
                    usage_block = None

                _add_usage(u=resp_obj.usage)

                history.append({"role": "assistant", 
                                "content": resp_obj.text})
                result = {"answer": resp_obj.text,
                          "display": None,
                          "history": history}
                if include_usage:
                    result["usage_turn"] = usage_block
                    result["usage_total"] = total_usage
                return result

            for _ in range(max_turns):
                resp_raw = await self.llm_generate_once(history=history, 
                                                        use_tools=True, 
                                                        tool_choice="auto", include_usage=include_usage)

                if include_usage:
                    resp_obj = resp_raw["response"]  # type: ignore[index]
                    usage_block = resp_raw["usage"]  # type: ignore[index]
                else:
                    resp_obj = resp_raw  # type: ignore[assignment]
                    usage_block = None

                _add_usage(u=resp_obj.usage)

                history.append({"role": "assistant",
                                "content": resp_obj.text,
                                **({"tool_calls": [tc.model_dump() for tc in resp_obj.tool_calls]} if resp_obj.tool_calls else {})})

                if not resp_obj.tool_calls:
                    result = {"answer": resp_obj.text, 
                              "display": None, 
                              "history": history}
                    if include_usage:
                        result["usage_turn"] = usage_block
                        result["usage_total"] = total_usage
                    return result

                early = await self.run_tool_calls(tool_calls=resp_obj.tool_calls, history=history)
                if early:
                    early["history"] = history
                    if include_usage:
                        early["usage_turn"] = usage_block
                        early["usage_total"] = total_usage
                    return early

            # Turn limit reached --------------------------------------------------
            result = {"answer": "[Turn limit reached]", 
                      "display": None, 
                      "history": history}
            if include_usage:
                result["usage_turn"] = None
                result["usage_total"] = total_usage
            return result
        finally:
            self._hist = None

    # ------------------------------------------------------------------ streaming chat (refactored Phase-4)

    async def stream_chat(self, 
                          user_query: str, 
                          history: List[dict], 
                          *, 
                          max_turns: int = 5, 
                          include_usage: bool = False) -> AsyncGenerator[ChatChunk, None]:  # noqa: D401
        if history is None:
            raise ValueError("history required")
        if self._hist is not None:
            raise RuntimeError("stream_chat re-entered")
        self._hist = history
        try:
            # ------------------------------------------------------------
            total_usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

            def _add_usage(u) -> None:  # noqa: D401
                if u is None:
                    return
                total_usage.prompt_tokens += u.prompt_tokens or 0
                total_usage.completion_tokens += u.completion_tokens or 0
                total_usage.total_tokens += u.total_tokens or 0

            self.prepare_history(user_query=user_query, 
                                 history=history)
            # LLM-only fast path --------------------------------------------------
            if not self.tools_enabled():
                usage_block = None
                async for chunk in self.llm_stream_once(history=history, use_tools=False, include_usage=include_usage):
                    if isinstance(chunk, str):
                        yield ChatChunk(type="llm_delta", text=chunk)
                    else:
                        # final chunk with usage (when include_usage=True)
                        if include_usage:
                            usage_block = chunk.get("usage")  # type: ignore[attr-defined]
                            _add_usage(u=usage_block)
                if include_usage:
                    yield ChatChunk(type="token_usage", 
                                    usage=usage_block, 
                                    cumulative=False)
                    yield ChatChunk(type="token_usage", 
                                    usage=total_usage, 
                                    cumulative=True)
                return

            for _ in range(max_turns):
                agg: List[str] = []
                final_resp: Optional[LLMResponse] = None
                usage_block = None

                async for c in self.llm_stream_once(history=history, 
                                                    use_tools=True, 
                                                    tool_choice="auto", 
                                                    include_usage=include_usage):
                    if isinstance(c, str):
                        agg.append(c)
                        yield ChatChunk(type="llm_delta", text=c)
                    else:
                        if include_usage:
                            usage_block = c.get("usage")  # type: ignore[attr-defined]
                            final_resp = c["response"]  # type: ignore[index]
                            _add_usage(usage_block)
                        else:
                            final_resp = c

                if final_resp is None:
                    final_resp = LLMResponse(model_id="unknown", 
                                             text="".join(agg))

                history.append({"role": "assistant",
                                "content": final_resp.text,
                                **({"tool_calls": [tc.model_dump() for tc in final_resp.tool_calls]} if final_resp.tool_calls else {})})

                if not final_resp.tool_calls:
                    if include_usage:
                        yield ChatChunk(type="token_usage", usage=usage_block, cumulative=False)
                        yield ChatChunk(type="token_usage", usage=total_usage, cumulative=True)
                    return

                early = await self.run_tool_calls(tool_calls=final_resp.tool_calls, history=history)
                if early:
                    if early.get("display"):
                        yield ChatChunk(type="tool_display", 
                                        envelope=early["display"])
                    if early.get("answer"):
                        yield ChatChunk(type="llm_delta", 
                                        text=early["answer"])
                    if include_usage:
                        yield ChatChunk(type="token_usage", usage=usage_block, cumulative=False)
                        yield ChatChunk(type="token_usage", usage=total_usage, cumulative=True)
                    return

            # Turn limit reached --------------------------------------------------
            yield ChatChunk(type="llm_delta", text="[Turn limit reached]")
            if include_usage:
                yield ChatChunk(type="token_usage", usage=total_usage, cumulative=True)
        finally:
            self._hist = None

    # ------------------------------------------------------------------ legacy test hooks
    async def _run_tool(self, tc: ToolCall):
        return await self.executor._run_tool(tool_call=tc, history=self._hist or [])

    def _compute_execution_layers(self, calls: List[ToolCall]):
        extra = getattr(self, "_tool_deps", {})  # type: ignore[attr-defined]
        reg = self.registry
        class _R:  # minimal proxy
            def depends_on(_, n):  # noqa: D401
                return set(reg.depends_on(name=n)) | set(extra.get(n, set()))
        tmp = Executor(pool=self.pool, registry=_R(), cache=NullCache())
        return tmp.compute_execution_layers(tool_calls=calls)

    # ------------------------------------------------------------------ auth

    def add_auth_provider(self, provider: BaseAuthProvider):  # noqa: D401
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

    def append_assistant_message(self, content: str, history: List[dict] | None = None, *, tool_calls: List[ToolCall] | None = None) -> List[dict]:  # noqa: D401
        """Append an *assistant* message to *history*.

        When called from inside :pyfunc:`chat` or :pyfunc:`stream_chat` the
        current internal history (``self._hist``) is used automatically.  For
        external callers pass the *history* list explicitly.  The helper keeps
        the message schema consistent with other parts of *AIClient* and
        serialises *tool_calls* via :pyfunc:`ToolCall.model_dump` for
        wire-compatibility.
        """
        # Resolve target history ------------------------------------------------
        target = history if history is not None else self._hist
        if target is None:
            raise ValueError("history required – either pass *history* or call within chat/stream_chat")

        entry: dict[str, Any] = {"role": "assistant",
                                 "content": content,}
        if tool_calls:
            entry["tool_calls"] = [tc.model_dump() for tc in tool_calls]

        target.append(entry)
        return target 