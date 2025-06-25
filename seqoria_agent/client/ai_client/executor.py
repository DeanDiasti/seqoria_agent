from __future__ import annotations

"""Execution utilities for *seqoria_agent*'s new AIClient implementation.

This module is introduced in **Phase 3 – Executor extraction** (see
``client-migration.md``) and encapsulates *all* logic required to run a set of
requested tool calls.

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
from typing import Any, Dict, List, Optional, Set, Tuple

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.client.ai_client.constants import (
    CLARIFY_TOOL_NAME,
    SUMMARIZE_TOOL_NAME,
    THINK_TOOL_NAME,
)
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.ai_client.utils import (
    extract_error_from_xml,
    null_async_cm,
    build_cache_key,
)
from seqoria_agent.client.ai_client.history import summarise_history
from seqoria_agent.client.ai_client.callbacks import (
    on_display as _on_display_cb,
    on_error as _on_error_cb,
    on_metadata as _on_metadata_cb,
)
from seqoria_agent.client.cache import CacheProtocol, NullCache
from seqoria_agent.client.parameters import ParamContext, ParameterProvider
from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.errors import CircuitOpenError, DependencyError, ToolRemoteError
from seqoria_agent.providers.schemas import ToolCall
from seqoria_agent.client.ai_client.consts_runtime import AI_MESSAGE_KEY  # noqa: F401

__all__: list[str] = [
    "Executor",
]


class Executor:  # noqa: D101 – docstring provided above
    # *Note*: keep implementation ≤ 350 LOC to satisfy migration gate.

    def __init__(
        self,
        *,
        pool: ConnectionPool,
        registry: ToolRegistry,
        cache: CacheProtocol | None = None,
        cache_ttl: int | None = None,
        tool_semaphore: Optional[asyncio.Semaphore] = None,
        tool_timeout: float | None = None,
        param_providers: Optional[List[ParameterProvider]] = None,
        allow_override: bool = False,
        on_display=None,
        on_error=None,
        on_metadata=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Core collaborators --------------------------------------------------
        self._pool = pool
        self._registry = registry

        # Caching -------------------------------------------------------------
        self._cache: CacheProtocol = cache or NullCache()  # type: ignore[assignment]
        self._cache_ttl = cache_ttl

        # Execution controls --------------------------------------------------
        self._tool_semaphore = tool_semaphore
        self._tool_timeout = tool_timeout

        # Parameter injection -------------------------------------------------
        self._param_providers = list(param_providers or [])
        self._allow_override = bool(allow_override)

        # Callbacks -----------------------------------------------------------
        self._on_display = on_display
        self._on_error = on_error
        self._on_metadata = on_metadata

        # Logger --------------------------------------------------------------
        self._logger = logger or logging.getLogger("seqoria_agent.executor")

    # ------------------------------------------------------------------
    # Public façade
    # ------------------------------------------------------------------

    async def run_calls(
        self,
        tool_calls: List[ToolCall],
        *,
        history: List[dict],
    ) -> Optional[dict[str, Any]]:
        """Execute *tool_calls* respecting dependencies.

        The *history* list is mutated in-place so callers see updates
        immediately.

        Returns a *payload* dict when an early-exit condition is reached
        (display-only or clarify tool).  Otherwise ``None`` is returned.
        """
        try:
            layers = self.compute_execution_layers(tool_calls)
        except DependencyError as de:
            # Surface dependency errors so callers can append an error answer.
            self._logger.error("Dependency resolution failed", error=str(de))
            return {"answer": f"[DependencyError] {de}", "display": None}

        early_exit_payload: Optional[dict[str, Any]] = None

        for layer in layers:
            coros = [self._run_tool(tc, history) for tc in layer]
            results = await asyncio.gather(*coros, return_exceptions=False)

            for (_, history_entry, maybe_payload) in results:
                if history_entry:
                    history.append(history_entry)

                if early_exit_payload is None and maybe_payload is not None:
                    early_exit_payload = maybe_payload

            if early_exit_payload is not None:
                return early_exit_payload

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_tool(
        self,
        tool_call: ToolCall,
        history: List[dict],
    ) -> Tuple[str | None, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Execute *tool_call* and return tuple
        ``(tool_call_id, history_entry, early_exit_payload)``.
        """
        func_dict = tool_call.function or {}
        tool_name = func_dict.get("name")
        try:
            arg_payload = func_dict.get("arguments", {})
            if isinstance(arg_payload, str):
                arguments: Dict[str, Any] = json.loads(arg_payload)
            elif isinstance(arg_payload, dict):
                arguments = arg_payload  # type: ignore[assignment]
            else:
                arguments = {}
        except Exception as e:  # noqa: BLE001
            arguments = {}
            self._logger.warning("Could not parse arguments for tool", tool=tool_name, error=str(e))

        # ------------------------------------------------------------------
        # Client-side server-populated parameter injection ------------------
        # ------------------------------------------------------------------
        server_args: Dict[str, Any] = {}

        spp_fields = self._registry.server_params(tool_name)
        if spp_fields:
            # Remove AI-provided values when override disallowed
            if not self._allow_override:
                for fld in spp_fields:
                    arguments.pop(fld, None)

            missing = [fld for fld in spp_fields if fld not in arguments]
            if missing and self._param_providers:
                ctx: ParamContext = {
                    "history": history,
                }
                for fld in list(missing):
                    for prov in self._param_providers:
                        try:
                            res = prov.resolve(  # type: ignore[attr-defined]
                                param_name=fld,
                                tool_name=tool_name,
                                context=ctx,
                            )
                            if asyncio.iscoroutine(res):
                                res = await res  # type: ignore[assignment]
                        except Exception:  # noqa: BLE001 – provider errors are ignored
                            continue
                        if res is not None:
                            server_args[fld] = res
                            break

            still_missing = [fld for fld in spp_fields if fld not in arguments and fld not in server_args]
            if still_missing:
                raise RuntimeError(
                    f"Missing server-populated parameters for tool '{tool_name}': {', '.join(still_missing)}"
                )

        # ------------------------------------------------------------------
        # Synthetic meta-tools ---------------------------------------------
        # ------------------------------------------------------------------
        if tool_name == THINK_TOOL_NAME:
            thought_text = arguments.get("thought", "")
            history_entry = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": thought_text,
            }
            self._logger.info("LLM thinking step recorded")
            return (tool_call.id, history_entry, None)

        if tool_name == SUMMARIZE_TOOL_NAME:
            summary_text = summarise_history(history)
            # Replace existing history (keep system prompt at idx 0)
            if history and history[0].get("role") == "system":
                history[1:] = [{"role": "assistant", "content": summary_text}]
            else:
                history[:] = [{"role": "assistant", "content": summary_text}]

            history_entry = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": "conversation summarized",
            }
            self._logger.info("Conversation summarized and history compressed")
            return (tool_call.id, history_entry, None)

        if tool_name == CLARIFY_TOOL_NAME:
            question_text = arguments.get("question", "")
            history_entry = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": question_text,
            }
            early_payload = {
                "answer": question_text,
                "display": None,
            }
            self._logger.info("Clarification requested from user")
            return (tool_call.id, history_entry, early_payload)

        # ------------------------------------------------------------------
        # Regular tool execution (with cache, timeout, breaker, …) ---------
        # ------------------------------------------------------------------
        cache_key = build_cache_key(tool_name=tool_name, arguments=arguments)
        show_user = arguments.get("show_user")
        is_display_call = self._registry.is_display_tool(tool_name) and bool(show_user)

        raw_output: Any | None = None
        try:
            cached_output = await self._cache.get(cache_key)
            if cached_output is not None:
                raw_output = cached_output
                self._logger.debug("Cache hit – skipping tool execution")
        except Exception as e:  # noqa: BLE001
            self._logger.warning("Cache error (ignored)", error=str(e))

        if raw_output is None:
            self._logger.debug("Executing tool", tool=tool_name, args=arguments)
            try:
                async with (self._tool_semaphore or null_async_cm()):
                    exec_coro = self._pool.call_tool(
                        name=tool_name,
                        arguments=arguments,
                        server_args=server_args,
                    )
                    if self._tool_timeout:
                        exec_coro = asyncio.wait_for(exec_coro, timeout=self._tool_timeout)
                    tool_result = await exec_coro
            except asyncio.TimeoutError:
                raw_output = f"Tool '{tool_name}' timed out after {self._tool_timeout}s"
            except CircuitOpenError as e:
                raw_output = f"Circuit open – {e}"
            except ToolRemoteError as e:
                raw_output = f"Error executing tool: {e}"
            else:
                if tool_result and hasattr(tool_result[0], "text"):
                    raw_output = tool_result[0].text  # type: ignore[assignment]
                else:
                    raw_output = "Tool executed successfully with no textual output."

                # Persist to cache ------------------------------------------------
                try:
                    await self._cache.set(cache_key, raw_output, ttl=self._cache_ttl)
                except Exception as e:  # noqa: BLE001
                    self._logger.warning("Cache set error (ignored)", error=str(e))

        # ------------------------------------------------------------------
        # Display envelope & structured metadata extraction ----------------
        # ------------------------------------------------------------------
        display_envelope: Optional[dict[str, Any]] = None
        metadata_block: Optional[dict[str, Any]] = None
        error_payload: Optional[dict[str, Any]] = None

        if isinstance(raw_output, (dict, list)):
            display_envelope = raw_output  # may be DisplayEnvelope later
        elif isinstance(raw_output, str):
            # Try JSON wrapper first
            try:
                tmp = json.loads(raw_output)
                if isinstance(tmp, dict) and AI_MESSAGE_KEY in tmp:
                    metadata_block = tmp.get("metadata", {})
                    raw_output = tmp.get(AI_MESSAGE_KEY, "")
            except Exception:  # noqa: BLE001
                pass

            # XML error wrapper -------------------------------------------------
            error_payload = extract_error_from_xml(xml_str=raw_output)

            # Attempt parsing into display envelope if JSON-like str
            if display_envelope is None:
                try:
                    display_envelope = json.loads(raw_output)
                except Exception:  # noqa: BLE001
                    pass

        # Attempt pydantic validation for DisplayEnvelope
        if display_envelope is not None and not isinstance(display_envelope, DisplayEnvelope):
            try:
                display_envelope = DisplayEnvelope.model_validate(display_envelope).dict_for_transport()
            except Exception:  # noqa: BLE001
                pass

        history_entry = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_name,
            "content": raw_output,
        }

        # Callback – metadata --------------------------------------------------
        if metadata_block is not None:
            _on_metadata_cb(metadata_block, history, self._on_metadata)

        # Display early-exit ---------------------------------------------------
        if is_display_call:
            answer_text = f"[Action '{tool_name}' was successful.]"
            early_payload = {
                "answer": answer_text,
                "display": display_envelope,
            }

            if display_envelope is not None:
                _on_display_cb(display_envelope, history, self._on_display)

            return (tool_call.id, history_entry, early_payload)

        # Structured error callback -------------------------------------------
        if error_payload is not None:
            _on_error_cb(error_payload, history, self._on_error)

        return (tool_call.id, history_entry, None)

    # ------------------------------------------------------------------
    # Dependency graph utilities (moved from AIClient)
    # ------------------------------------------------------------------

    def compute_execution_layers(self, tool_calls: List[ToolCall]) -> List[List[ToolCall]]:
        """Return topologically-sorted execution layers for *tool_calls*."""
        call_map: Dict[str, ToolCall] = {
            (tc.function or {}).get("name"): tc for tc in tool_calls
        }
        requested: Set[str] = set(call_map.keys())

        # Build graph (edges: dep → tool)
        graph: Dict[str, Set[str]] = defaultdict(set)
        indegree: Dict[str, int] = {name: 0 for name in requested}

        for tool_name in requested:
            deps = self._registry.depends_on(tool_name)
            for dep in deps:
                if dep in requested:
                    graph[dep].add(tool_name)
                    indegree[tool_name] += 1

        # Kahn — produce layers ----------------------------------------------
        layers: List[List[ToolCall]] = []
        queue: deque[str] = deque([n for n, d in indegree.items() if d == 0])
        processed = 0

        while queue:
            current_layer: List[ToolCall] = []
            for _ in range(len(queue)):
                n = queue.popleft()
                processed += 1
                current_layer.append(call_map[n])
                for succ in graph.get(n, []):
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        queue.append(succ)
            layers.append(current_layer)

        if processed != len(requested):
            raise DependencyError("Cyclic dependency detected between requested tool calls.")

        return layers

    # ------------------------------------------------------------------
    # Helper – naive summarisation (copied from AIClient)
    # ------------------------------------------------------------------
    # summarisation helper moved to seqoria_agent.client.ai_client.history 