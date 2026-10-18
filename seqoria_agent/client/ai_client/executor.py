from __future__ import annotations

"""Dependency-aware concurrent tool execution, cancellation, caching, and parameter injection."""

import asyncio
import copy
import uuid
import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Union, AsyncIterator, Tuple, Mapping

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.ai_client.callbacks import (
    on_error as _on_error_cb,
    on_metadata as _on_metadata_cb,
)
from seqoria_agent.client.ai_client.utils import json_loads, build_cache_key
from seqoria_agent.client.cache import CacheProtocol, NullCache
from seqoria_agent.client.parameters import ParameterRegistry, ParameterSource, ParamContext
from seqoria_agent.models.errors import DependencyError
from seqoria_agent.client.ai_client.consts_runtime import AI_MESSAGE_KEY
from seqoria_agent.models.chunk import (
    ToolCallResult,
    ToolCallError,
    ToolCallRequest,
    ToolCallRequestList,
)

__all__: list[str] = ["Executor"]


class Executor:
    def __init__(
        self,
        *,
        pool: MCPConnectionPool,
        registry: ToolRegistry,
        cache: CacheProtocol | None = None,
        cache_ttl: int | None = None,
        tool_semaphore: Optional[asyncio.Semaphore] = None,
        tool_timeout: float | None = None,
        param_registry: Optional[Union[Mapping[str, ParameterSource], ParameterRegistry]] = None,
        allow_override: bool = False,
        on_error=None,
        on_metadata=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Core collaborators --------------------------------------------------
        self._pool = pool
        self._registry = registry

        # Caching -------------------------------------------------------------
        self._cache: CacheProtocol = cache or NullCache()
        self._cache_ttl = cache_ttl
        self._cache_namespace = uuid.uuid4().hex

        # Execution controls --------------------------------------------------
        self._tool_semaphore = tool_semaphore or asyncio.Semaphore(value=16)
        self._tool_timeout = tool_timeout

        # Parameter injection -------------------------------------------------
        if isinstance(param_registry, ParameterRegistry):
            self._param_registry = param_registry
        else:
            self._param_registry = ParameterRegistry(mapping=param_registry or {})
        self._allow_override = bool(allow_override)

        # Callbacks -----------------------------------------------------------
        self._on_error = on_error
        self._on_metadata = on_metadata

        # Logger --------------------------------------------------------------
        self._logger = logger or logging.getLogger(name="seqoria_agent.executor")

    async def run_call_stream(
        self,
        tool_calls: Union[List[ToolCallRequest], ToolCallRequestList],
        *,
        context: ParamContext | None = None,
    ) -> AsyncIterator[Tuple[ToolCallRequest, ToolCallResult]]:
        """Yield :class:`ToolCallResult` objects as soon as they are ready.

        The generator respects *dependency layers* computed by
        :pyfunc:`compute_execution_layers`.  All tools within a layer start
        concurrently (subject to the executor semaphore) and results are
        yielded in the order they *finish*.  The next layer only starts once
        every task of the current layer has completed, guaranteeing that
        dependency prerequisites have run first.
        """
        # Normalise convenience wrapper types ---------------------------------
        if isinstance(tool_calls, ToolCallRequestList):
            tool_calls = tool_calls.tool_calls

        ctx: ParamContext = context or {}

        # Single-call fast-path ------------------------------------------------
        if len(tool_calls) == 1:
            single_req = tool_calls[0]
            yield (single_req, await self._run_tool(tool_call=single_req, context=ctx))  # type: ignore[arg-type]
            return

        # Multi-call with dependency handling ---------------------------------
        try:
            layers = self.compute_execution_layers(tool_calls=tool_calls)
        except DependencyError as de:
            # Surface dependency errors via logger and abort stream.
            self._logger.error("Dependency resolution failed: %s", de)
            for tc in tool_calls:
                yield tc, ToolCallError(tool_call_id=tc.id, name=tc.name, error=str(de))
            return

        # ------------------------------------------------------------------
        # New failure-propagation logic
        # ------------------------------------------------------------------

        failed_names: Set[str] = set()
        deps_cache: Dict[str, Set[str]] = {}

        for layer in layers:
            # --------------------------------------------------------------
            # Partition the layer into *blocked* (dependency failed) and
            # *ready* tool calls.
            # --------------------------------------------------------------
            blocked_calls: List[ToolCallRequest] = []
            ready_calls: List[ToolCallRequest] = []

            for tc in layer:
                deps = deps_cache.get(tc.name)
                if deps is None:
                    deps = self._registry.depends_on(name=tc.name)
                    deps_cache[tc.name] = deps
                if deps & failed_names:
                    blocked_calls.append(tc)
                else:
                    ready_calls.append(tc)

            # Track per-name success flags for this layer so we can decide
            # which tool names "globally" failed (all calls resulted in
            # errors or were skipped).
            per_name_success: Dict[str, List[bool]] = defaultdict(list)

            # --------------------------------------------------------------
            # Emit skipped errors for blocked calls *before* executing ready
            # calls (so consumers see the skip ASAP).
            # --------------------------------------------------------------
            for tc in blocked_calls:
                deps = deps_cache.get(tc.name)
                if deps is None:
                    deps = self._registry.depends_on(name=tc.name)
                    deps_cache[tc.name] = deps
                bad_deps = sorted(deps & failed_names)
                err_msg = (
                    ("Skipped because dependency failure: " + ", ".join(bad_deps))
                    if bad_deps
                    else "Skipped due to dependency failure"
                )

                if getattr(self._logger, "isEnabledFor", lambda *_: True)(logging.INFO):
                    self._logger.info(
                        "Skipping tool %s due to failed dependencies: %s", tc.name, bad_deps
                    )

                error_entry = ToolCallError(tool_call_id=tc.id, name=tc.name, error=err_msg)

                per_name_success[tc.name].append(False)
                yield (tc, error_entry)

            # --------------------------------------------------------------
            # Execute *ready* tool calls concurrently and stream back their
            # results as they finish.
            # --------------------------------------------------------------
            if ready_calls:

                async def _exec_pair(tc: ToolCallRequest) -> Tuple[ToolCallRequest, ToolCallResult]:  # noqa: D401
                    res = await self._run_tool(tool_call=tc, context=ctx)
                    return tc, res

                tasks = [asyncio.create_task(coro=_exec_pair(tc=tc)) for tc in ready_calls]

                try:
                    for fut in asyncio.as_completed(fs=tasks):
                        tc, res = await fut
                        per_name_success[tc.name].append(not isinstance(res, ToolCallError))
                        yield (tc, res)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            # --------------------------------------------------------------
            # Post-layer aggregation – mark a tool *name* as failed when
            # **all** of its calls in this layer failed (or were skipped).
            # --------------------------------------------------------------
            for name, flags in per_name_success.items():
                if not any(flags):
                    failed_names.add(name)

    # ----------------------------------------------------------------------
    # Backwards-compatibility wrapper – accumulate results into a list
    # ----------------------------------------------------------------------
    async def run_calls(
        self,
        tool_calls: Union[List[ToolCallRequest], ToolCallRequestList],
        *,
        context: ParamContext | None = None,
    ) -> List[ToolCallResult]:
        """Execute *tool_calls* and return a list of results.

        This is now implemented on top of :pyfunc:`run_call_stream` to avoid
        duplicating the execution logic while preserving the original API.
        """
        calls = tool_calls.tool_calls if isinstance(tool_calls, ToolCallRequestList) else tool_calls
        # Streaming is completion-ordered; callers of run_calls pair by input order.
        results = {
            id(req): res
            async for req, res in self.run_call_stream(tool_calls=calls, context=context or {})
        }
        return [results[id(req)] for req in calls]

    async def _run_tool(
        self, tool_call: ToolCallRequest, *, context: ParamContext
    ) -> ToolCallResult:
        """Execute *tool_call* and return a fully-formed chat *history* entry.

        No tuple or additional metadata is returned - early-exit or metadata
        handling is performed via callbacks.
        """
        tool_name = tool_call.name
        arguments = copy.deepcopy(tool_call.arguments or {})

        # Client-side server-populated parameter injection ------------------
        server_args: Dict[str, Any] = {}

        spp_fields = self._registry.server_params(name=tool_name)
        if spp_fields:
            # Remove AI-provided values when override disallowed
            if not self._allow_override:
                for fld in spp_fields:
                    arguments.pop(fld, None)

            missing = [fld for fld in spp_fields if fld not in arguments]
            for fld in list(missing):
                try:
                    res = await self._param_registry.resolve(name=fld, context=context)
                except Exception:  # pragma: no cover – safety guard
                    res = None
                if res is not None:
                    server_args[fld] = res

            still_missing = [
                fld for fld in spp_fields if fld not in arguments and fld not in server_args
            ]
            if still_missing:
                raise RuntimeError(
                    f"Missing server-populated parameters for tool '{tool_name}': {', '.join(still_missing)}"
                )

        # --------------------------------------------------------------------
        # Client-side caching (best-effort)
        #
        # NOTE: Cache is opt-in by configuration (default is NullCache).
        # We include server_args in the key to avoid cross-tenant / cross-user
        # leakage when server-populated parameters affect results.
        # --------------------------------------------------------------------

        cache_key: str | None = None
        if not isinstance(self._cache, NullCache):
            try:
                # Include the whole trusted request context: permissions can vary
                # between users in the same tenant and between requests by one user.
                merged_for_cache = {
                    "client": self._cache_namespace,
                    "registry_version": self._registry.version,
                    "arguments": arguments,
                    "server_args": server_args,
                    "context": dict(context),
                }
                cache_key = build_cache_key(tool_name=tool_name, arguments=merged_for_cache)
            except (TypeError, ValueError):
                # Unserializable context cannot be safely cached.
                cache_key = None

        raw_output: Any | None = None
        if cache_key is not None:
            try:
                cached = await self._cache.get(cache_key)
            except Exception:  # pragma: no cover – cache backend must not break execution
                cached = None
            if cached is not None:
                raw_output = (
                    cached.decode("utf-8") if isinstance(cached, (bytes, bytearray)) else cached
                )

        # Regular tool execution (with timeout, circuit-breaker, …)
        if getattr(self._logger, "isEnabledFor", lambda *_: True)(logging.DEBUG):
            self._logger.debug("Executing tool %s", tool_name)

        if raw_output is None:
            try:
                async with self._tool_semaphore:
                    exec_coro = self._pool.call_tool(
                        name=tool_name, arguments=arguments, server_args=server_args
                    )
                    if self._tool_timeout:
                        exec_coro = asyncio.wait_for(fut=exec_coro, timeout=self._tool_timeout)
                    tool_result = await exec_coro

                # --------------------------- success path ---------------------------
                if tool_result and hasattr(tool_result[0], "text"):
                    raw_output = tool_result[0].text  # type: ignore[assignment]
                else:
                    raw_output = "Tool executed successfully with no textual output."

                # Best-effort cache write (only on success)
                if cache_key is not None and isinstance(raw_output, str):
                    try:
                        await self._cache.set(cache_key, raw_output, ttl=self._cache_ttl)
                    except Exception:  # pragma: no cover – cache backend must not break execution
                        pass

            except Exception as exc:
                # -------------------------------------------------------------------
                # Failure path – encapsulate the error in a ToolCallError instance so
                # callers can react without relying on side-band callbacks.
                # -------------------------------------------------------------------
                err_msg = (
                    f"Tool '{tool_name}' failed: {exc}"
                    if not isinstance(exc, asyncio.TimeoutError)
                    else f"Tool '{tool_name}' timed out after {self._tool_timeout}s"
                )

                # Structured callback so external apps can log/telemetry the error
                _on_error_cb(
                    error_payload={
                        "tool": tool_name,
                        "tool_call_id": tool_call.id,
                        "message": err_msg,
                    },
                    cb=self._on_error,
                )

                return ToolCallError(tool_call_id=tool_call.id, name=tool_name, error=err_msg)
        # --------------------------------------------------------------------

        # Display envelope & structured metadata extraction ----------------
        metadata_block: Optional[dict[str, Any]] = None

        if isinstance(raw_output, str):
            # Try JSON wrapper first
            # Guard against exception-heavy parsing when output is clearly not a JSON envelope.
            if AI_MESSAGE_KEY in raw_output:
                stripped = raw_output.lstrip()
                if stripped.startswith("{"):
                    try:
                        tmp = json_loads(raw_output)
                        if isinstance(tmp, dict) and AI_MESSAGE_KEY in tmp:
                            metadata_block = tmp.get("metadata", {})
                            raw_output = tmp.get(AI_MESSAGE_KEY, "")
                    except Exception:  # noqa: BLE001
                        pass

        history_entry = ToolCallResult(
            tool_call_id=tool_call.id,
            role="tool_result",
            name=tool_name,
            type="tool_use",
            result=raw_output,
        )

        # Callback – metadata --------------------------------------------------
        if metadata_block is not None:
            _on_metadata_cb(meta=metadata_block, cb=self._on_metadata)

        return history_entry

    def compute_execution_layers(
        self, tool_calls: List[ToolCallRequest]
    ) -> List[List[ToolCallRequest]]:
        """Return topologically-sorted execution layers for *tool_calls*."""

        # Group calls by *name* while preserving original order of duplicates
        name_to_calls: Dict[str, List[ToolCallRequest]] = defaultdict(list)
        for tc in tool_calls:
            tool_name = tc.name
            name_to_calls[tool_name].append(tc)

        requested_names: Set[str] = set(name_to_calls.keys())

        # Build dependency graph (edges: dep_name → tool_name)
        graph: Dict[str, Set[str]] = defaultdict(set)
        indegree: Dict[str, int] = {name: 0 for name in requested_names}

        deps_cache: Dict[str, Set[str]] = {}
        for tool_name in requested_names:
            deps = deps_cache.get(tool_name)
            if deps is None:
                deps = self._registry.depends_on(name=tool_name)
                deps_cache[tool_name] = deps
            for dep in deps:
                if dep in requested_names:
                    graph[dep].add(tool_name)
                    indegree[tool_name] += 1

        # Topological sort (Kahn) – but expand duplicate calls per node
        layers: List[List[ToolCallRequest]] = []
        queue: deque[str] = deque(iterable=[n for n, d in indegree.items() if d == 0])
        processed = 0

        while queue:
            current_layer: List[ToolCallRequest] = []
            for _ in range(len(queue)):
                n = queue.popleft()
                processed += 1
                # Append *all* calls for this tool name – preserves duplicates
                current_layer.extend(name_to_calls[n])
                for succ in graph.get(n, []):
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        queue.append(succ)
            layers.append(current_layer)

        if processed != len(requested_names):
            raise DependencyError("Cyclic dependency detected between requested tool calls.")

        return layers
