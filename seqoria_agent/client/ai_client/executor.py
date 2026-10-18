"""Dependency-aware concurrent tool execution, cancellation, and parameter injection."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
from collections import defaultdict, deque
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Set, Tuple, Union

from seqoria_agent._redaction import redact_text
from seqoria_agent.client.ai_client.callbacks import on_error as _on_error_cb
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.parameters import ParamContext, ParameterRegistry, ParameterSource
from seqoria_agent.models.chunk import (
    ToolCallError,
    ToolCallRequest,
    ToolCallRequestList,
    ToolCallResult,
)
from seqoria_agent.models.errors import DependencyError

__all__: list[str] = ["Executor"]


class Executor:
    def __init__(
        self,
        *,
        pool: MCPConnectionPool,
        registry: ToolRegistry,
        tool_semaphore: Optional[asyncio.Semaphore] = None,
        tool_timeout: float | None = None,
        param_registry: Optional[Union[Mapping[str, ParameterSource], ParameterRegistry]] = None,
        allow_override: bool = False,
        on_error=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Core collaborators --------------------------------------------------
        self._pool = pool
        self._registry = registry

        # Execution controls --------------------------------------------------
        if tool_timeout is not None and (
            not isinstance(tool_timeout, (int, float))
            or isinstance(tool_timeout, bool)
            or not math.isfinite(tool_timeout)
            or tool_timeout <= 0
        ):
            raise ValueError("tool_timeout must be positive and finite")
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

        # Logger --------------------------------------------------------------
        self._logger = logger or logging.getLogger(name="seqoria_agent.executor")

    async def run_call_stream(
        self,
        tool_calls: Union[List[ToolCallRequest], ToolCallRequestList],
        *,
        context: ParamContext | None = None,
    ) -> AsyncGenerator[Tuple[ToolCallRequest, ToolCallResult | ToolCallError], None]:
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

        # Multi-call with dependency handling ---------------------------------
        try:
            layers = self.compute_execution_layers(tool_calls=tool_calls)
        except DependencyError as de:
            # Surface dependency errors via logger and abort stream.
            self._logger.error("Dependency resolution failed: %s", redact_text(de))
            for tc in tool_calls:
                yield tc, ToolCallError(tool_call_id=tc.id, name=tc.name, error=redact_text(de))
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

                async def _exec_pair(
                    tc: ToolCallRequest,
                ) -> Tuple[ToolCallRequest, ToolCallResult | ToolCallError]:  # noqa: D401
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
    # Collect results in request order
    # ----------------------------------------------------------------------
    async def run_calls(
        self,
        tool_calls: Union[List[ToolCallRequest], ToolCallRequestList],
        *,
        context: ParamContext | None = None,
    ) -> List[ToolCallResult | ToolCallError]:
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
    ) -> ToolCallResult | ToolCallError:
        """Execute *tool_call* and return a fully-formed chat *history* entry.

        The shared concurrency limit covers parameter resolution
        and tool invocation. The timeout starts after acquiring a slot, so time
        spent waiting for other requests does not consume the execution budget.
        """
        try:
            async with self._tool_semaphore:
                async with asyncio.timeout(self._tool_timeout):
                    return await self._execute_tool(tool_call, context=context)
        except Exception as exc:
            # Parameter resolution and output processing are part of this call
            # too. One malformed request must not cancel unrelated tool tasks.
            return self._tool_error(tool_call, exc)

    def _tool_error(self, tool_call: ToolCallRequest, exc: Exception) -> ToolCallError:
        err_msg = (
            f"Tool '{tool_call.name}' timed out after {self._tool_timeout}s"
            if isinstance(exc, asyncio.TimeoutError)
            else f"Tool '{tool_call.name}' failed: {redact_text(exc)}"
        )
        _on_error_cb(
            error_payload={
                "tool": tool_call.name,
                "tool_call_id": tool_call.id,
                "message": err_msg,
            },
            cb=self._on_error,
        )
        return ToolCallError(tool_call_id=tool_call.id, name=tool_call.name, error=err_msg)

    async def _execute_tool(
        self, tool_call: ToolCallRequest, *, context: ParamContext
    ) -> ToolCallResult | ToolCallError:
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

        if getattr(self._logger, "isEnabledFor", lambda *_: True)(logging.DEBUG):
            self._logger.debug("Executing tool %s", tool_name)

        tool_result = await self._pool.call_tool(
            name=tool_name, arguments=arguments, server_args=server_args
        )
        text_blocks = [block.text for block in (tool_result or []) if hasattr(block, "text")]
        raw_output = (
            "\n".join(text_blocks)
            if text_blocks
            else "Tool executed successfully with no textual output."
        )

        history_entry = ToolCallResult(
            tool_call_id=tool_call.id,
            role="tool_result",
            name=tool_name,
            type="tool_use",
            result=raw_output,
        )

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
        indegree: Dict[str, int] = {name: 0 for name in name_to_calls}

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
                for succ in sorted(graph.get(n, [])):
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        queue.append(succ)
            layers.append(current_layer)

        if processed != len(requested_names):
            raise DependencyError("Cyclic dependency detected between requested tool calls.")

        return layers
