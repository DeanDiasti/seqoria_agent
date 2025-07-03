from __future__ import annotations

"""Execution utilities for *seqoria_agent*'s new AIClient implementation.

The public façade is the :class:`Executor` which receives already parsed
:class:`~seqoria_agent.providers.schemas.ToolCall` objects (as returned by the
LLM provider) together with the current conversation *history* list.  The
Executor then

1. Builds a dependency graph (via the moved ``_compute_execution_layers`` helper)
2. Executes tools in topological batches with configurable concurrency limits
3. Handles client-side cache lookup / storage
4. Applies timeout, retry & circuit-breaker protections delegated to
   :class:`~seqoria_agent.client.ai_client.connection_pool.ConnectionPool`
5. Implements synthetic meta-tools (``__think__`` / ``__summarize__`` /
   ``__clarify__``)
6. Detects early-exit display envelopes and surfaces structured error metadata
   via the provided callbacks.

The class purposefully stays agnostic of the surrounding *AIClient* orchestra –
all external interaction happens via the injected collaborators passed to the
constructor.  This design keeps the surface area minimal and makes the Executor
straight-forward to unit-test in isolation.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Union

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.client.ai_client.constants import THINK_TOOL_NAME
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.ai_client.callbacks import (on_error as _on_error_cb,
                                                      on_metadata as _on_metadata_cb)
from seqoria_agent.client.cache import CacheProtocol, NullCache
from seqoria_agent.client.parameters import ParameterProvider
from seqoria_agent.models.errors import CircuitOpenError, DependencyError, ToolRemoteError
from seqoria_agent.client.ai_client.consts_runtime import AI_MESSAGE_KEY 
from seqoria_agent.models.chunk import ToolCallResult, ToolCallRequest, ToolCallRequestList

__all__: list[str] = ["Executor"]


class Executor:

    def __init__(self,
                 *,
                 pool: ConnectionPool,
                 registry: ToolRegistry,
                 cache: CacheProtocol | None = None,
                 cache_ttl: int | None = None,
                 tool_semaphore: Optional[asyncio.Semaphore] = None,
                 tool_timeout: float | None = None,
                 param_providers: Optional[List[ParameterProvider]] = None,
                 allow_override: bool = False,
                 on_error=None,
                 on_metadata=None,
                 logger: Optional[logging.Logger] = None) -> None:
        # Core collaborators --------------------------------------------------
        self._pool = pool
        self._registry = registry

        # Caching -------------------------------------------------------------
        self._cache: CacheProtocol = cache or NullCache()
        self._cache_ttl = cache_ttl

        # Execution controls --------------------------------------------------
        self._tool_semaphore = tool_semaphore or asyncio.Semaphore(value=16)
        self._tool_timeout = tool_timeout

        # Parameter injection -------------------------------------------------
        self._param_providers = list(param_providers or [])
        self._allow_override = bool(allow_override)

        # Callbacks -----------------------------------------------------------
        self._on_error = on_error
        self._on_metadata = on_metadata

        # Logger --------------------------------------------------------------
        self._logger = logger or logging.getLogger(name="seqoria_agent.executor")

    async def run_calls(self,
                        tool_calls: Union[List[ToolCallRequest], ToolCallRequestList]) -> List[ToolCallResult]:
        """Execute *tool_calls* respecting dependencies and return chat history entries.

        The returned list contains one history entry per *ToolCall* in the same
        execution order that callers can directly append to their *history*
        list via ``history.extend(...)``.  The helper hides concurrency,
        dependency ordering and error handling details.
        """
        # Fast-path: single call – skip dependency sorting entirely
        if isinstance(tool_calls, ToolCallRequestList):
            tool_calls = tool_calls.tool_calls

        if len(tool_calls) == 1:
            # Always return a *list* so callers can blindly extend chat history
            t = [await self._run_tool(tool_call=tool_calls[0])]  # type: ignore[arg-type]
            
            return t

        # Multi-call execution with dependency handling
        try:
            layers = self.compute_execution_layers(tool_calls=tool_calls)
        except DependencyError as de:
            # Surface dependency errors so callers can append an error answer.
            self._logger.error(msg="Dependency resolution failed", 
                               error=str(object=de))
            return []
        results = []
        for layer in layers:
            coros = [self._run_tool(tool_call=tc) for tc in layer]
            results.extend(await asyncio.gather(*coros, return_exceptions=False))

        return results
    
    # Comment: Do we need to return the updated history or is the object updated and we don't need to return it?

    async def _run_tool(self,
                        tool_call: ToolCallRequest) -> ToolCallResult:
        """Execute *tool_call* and return a fully-formed chat *history* entry.

        No tuple or additional metadata is returned - early-exit or metadata
        handling is performed via callbacks.
        """
        tool_name = tool_call.name
        arguments = tool_call.arguments

        # Client-side server-populated parameter injection ------------------
        server_args: Dict[str, Any] = {}

        spp_fields = self._registry.server_params(name=tool_name)
        if spp_fields:
            # Remove AI-provided values when override disallowed
            if not self._allow_override:
                for fld in spp_fields:
                    arguments.pop(fld, None)

            missing = [fld for fld in spp_fields if fld not in arguments]
            if missing and self._param_providers:
                for fld in list(missing):
                    for prov in self._param_providers:
                        try:
                            res = prov.resolve(param_name=fld, 
                                               tool_name=tool_name)
                            if asyncio.iscoroutine(obj=res):
                                res = await res  # type: ignore[assignment]
                        except Exception:  # noqa: BLE001 – provider errors are ignored
                            continue
                        if res is not None:
                            server_args[fld] = res
                            break

            still_missing = [fld for fld in spp_fields if fld not in arguments and fld not in server_args]
            if still_missing:
                raise RuntimeError(msg=f"Missing server-populated parameters for tool '{tool_name}': {', '.join(still_missing)}")

        # Synthetic meta-tools
        if tool_name == THINK_TOOL_NAME:
            thought_text = f"<think>\n{arguments.get('thought', '').strip()}\n</think>"
            history_entry = ToolCallResult(tool_call_id=tool_call.id,
                                           role="tool_result",
                                           type="tool_use",
                                           name=tool_name,
                                           result=thought_text)
            if getattr(self._logger, "isEnabledFor", lambda *_: True)(logging.DEBUG):
                self._logger.info(msg="LLM thinking step recorded")
            return history_entry
        
        # Regular tool execution (with timeout, circuit-breaker, …)
        raw_output: Any | None = None

        if getattr(self._logger, "isEnabledFor", lambda *_: True)(logging.DEBUG):
            self._logger.debug(msg="Executing tool", tool=tool_name, args=arguments)

        try:
            async with self._tool_semaphore:
                exec_coro = self._pool.call_tool(name=tool_name,
                                                 arguments=arguments,
                                                 server_args=server_args)
                if self._tool_timeout:
                    exec_coro = asyncio.wait_for(fut=exec_coro,
                                                 timeout=self._tool_timeout)
                tool_result = await exec_coro
        except asyncio.TimeoutError:
            raw_output = f"Tool '{tool_name}' timed out after {self._tool_timeout}s"
        except CircuitOpenError as e:
            raw_output = f"Circuit open - {e}"
        except ToolRemoteError as e:
            raw_output = f"Error executing tool: {e}"
        else:
            if tool_result and hasattr(tool_result[0], "text"):
                raw_output = tool_result[0].text  # type: ignore[assignment]
            else:
                raw_output = "Tool executed successfully with no textual output."

        # Display envelope & structured metadata extraction ----------------
        metadata_block: Optional[dict[str, Any]] = None
        error_payload: Optional[dict[str, Any]] = None

        if isinstance(raw_output, str):
            # Try JSON wrapper first
            try:
                tmp = json.loads(s=raw_output)
                if isinstance(tmp, dict) and AI_MESSAGE_KEY in tmp:
                    metadata_block = tmp.get("metadata", {})
                    raw_output = tmp.get(AI_MESSAGE_KEY, "")
            except Exception:  # noqa: BLE001
                pass


        history_entry = ToolCallResult(tool_call_id=tool_call.id,
                                       role="tool_result",
                                       name=tool_name,
                                       type="tool_use",
                                       result=raw_output)

        # Callback – metadata --------------------------------------------------
        if metadata_block is not None:
            _on_metadata_cb(meta=metadata_block, 
                            cb=self._on_metadata)

        # Structured error callback -------------------------------------------
        if error_payload is not None:
            _on_error_cb(error_payload=error_payload, 
                         cb=self._on_error)

        return history_entry

    def compute_execution_layers(self, tool_calls: List[ToolCallRequest]) -> List[List[ToolCallRequest]]:
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

        for tool_name in requested_names:
            deps = self._registry.depends_on(name=tool_name)
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
