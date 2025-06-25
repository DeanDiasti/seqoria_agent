from __future__ import annotations

"""Phase-6 AIClient orchestrator (≤150 LOC).
Composes ConnectionPool, ToolRegistry, Executor and LLMInterface while
keeping the public surface identical to the legacy implementation.
"""

import asyncio, logging
from typing import Any, AsyncGenerator, Iterable, List, Optional, Self

try:  # Prefer structlog but fall back to stdlib
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None  # type: ignore

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.history import build_system_prompt
from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.cache import CacheProtocol, NullCache  # noqa: F401
from seqoria_agent.client.parameters import ParameterProvider
from seqoria_agent.models.chunk import ChatChunk
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall
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
        settings: ClientSettings | None = None,
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
            self._logger.warning("AIClient initialised without tools – LLM-only mode")
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
        # keep legacy behaviour using overridden _update_llm_tools when subclassed
        await self._update_llm_tools()

    def _tools_enabled(self) -> bool:
        return bool(self._tool_map or self._boxes)

    # ------------------------------------------------------------------ chat
    async def chat(self, user_query: str, history: List[dict], *, max_turns: int = 5):
        if history is None:
            raise ValueError("history required")
        if self._hist is not None:
            raise RuntimeError("chat re-entered")
        self._hist = history
        try:
            if not history or history[0].get("role") != "system":
                history.insert(0, {"role": "system", "content": self._system_prompt})
            history.append({"role": "user", "content": user_query})
            if not self._tools_enabled():
                r = await self.llm.generate(history=history)
                history.append({"role": "assistant", "content": r.text})
                return {"answer": r.text, "display": None, "history": history}
            for _ in range(max_turns):
                r = await self.llm.generate(history=history, tools=self.registry.llm_tools, tool_choice="auto")
                history.append({
                    "role": "assistant",
                    "content": r.text,
                    **({"tool_calls": [tc.model_dump() for tc in r.tool_calls]} if r.tool_calls else {}),
                })
                if not r.tool_calls:
                    return {"answer": r.text, "display": None, "history": history}
                early = await self.executor.run_calls(tool_calls=r.tool_calls, history=history)
                if early:
                    early["history"] = history
                    return early
            return {"answer": "[Turn limit reached]", "display": None, "history": history}
        finally:
            self._hist = None

    # ------------------------------------------------------------------ streaming chat
    async def stream_chat(self, user_query: str, history: List[dict], *, max_turns: int = 5) -> AsyncGenerator[ChatChunk, None]:
        if history is None:
            raise ValueError("history required")
        if self._hist is not None:
            raise RuntimeError("stream_chat re-entered")
        self._hist = history
        try:
            if not history or history[0].get("role") != "system":
                history.insert(0, {"role": "system", "content": self._system_prompt})
            history.append({"role": "user", "content": user_query})
            if not self._tools_enabled():
                gen = await self._providers[0].async_stream(chat_history=history)
                async for d in gen:
                    if isinstance(d, str):
                        yield ChatChunk(type="llm_delta", text=d)
                return
            for _ in range(max_turns):
                agg: List[str] = []
                final: Optional[LLMResponse] = None
                async for c in self.llm.stream(history=history, tools=self.registry.llm_tools, tool_choice="auto"):
                    if isinstance(c, str):
                        agg.append(c); yield ChatChunk(type="llm_delta", text=c)
                    else:
                        final = c
                if final is None:
                    final = LLMResponse(model_id="unknown", text="".join(agg))
                history.append({
                    "role": "assistant",
                    "content": final.text,
                    **({"tool_calls": [tc.model_dump() for tc in final.tool_calls]} if final.tool_calls else {}),
                })
                early = await self.executor.run_calls(tool_calls=final.tool_calls, history=history)
                if early:
                    if early.get("display"):
                        yield ChatChunk(type="tool_display", envelope=early["display"])
                    if early.get("answer"):
                        yield ChatChunk(type="llm_delta", text=early["answer"])
                    return
            yield ChatChunk(type="llm_delta", text="[Turn limit reached]")
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