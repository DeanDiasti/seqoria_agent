from __future__ import annotations

"""Central catalogue of tool metadata for AIClient.

This class is responsible for discovering tools from remote FastMCP servers
(via an existing :class:`~seqoria_agent.client.ai_client.connection_pool.ConnectionPool`)
**and** local :class:`~seqoria_agent.toolbox.LocalToolbox` instances, merging the
information, and exposing convenient accessors for other components.

It moves the functionality previously implemented inside
``seqoria_agent.client.client.AIClient._update_llm_tools`` so that the
monolithic AIClient can be refactored into smaller, testable pieces.
"""

# ruff clean imports
import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Set

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.ai_client.constants import (
    THINK_TOOL_NAME,
    CLARIFY_TOOL_NAME,
    SUMMARIZE_TOOL_NAME,
)

__all__: list[str] = [
    "ToolRegistry",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Maintain merged tool metadata originating from remote & local sources.

    Parameters
    ----------
    pool:
        An initialised :class:`ConnectionPool` instance used to retrieve the
        remote tool catalogue via ``client.list_tools()`` on each connection.
    local_toolboxes:
        A list of :class:`LocalToolbox` instances whose inline tools are merged
        with the remote catalog.
    tool_map:
        **Mutable** mapping that resolves ``tool_name → transport``.  The
        registry writes into the provided dict during :pyfunc:`refresh` so
        callers (e.g. :pyclass:`AIClient`) can look-up where to dispatch a tool
        call. *Not* copying the dict ensures both sides share a single source
        of truth.
    refresh_interval:
        Interval in **seconds** for automatic background refresh.  ``0``
        disables the refresher.
    logger:
        Optional :pymod:`logging` compatible logger.  A module-level logger is
        used when omitted.
    enable_meta_tools:
        Whether to include built-in meta-tools in the tool catalogue.
    """

    # Limit total LOC to stay within phase requirement (< 300 LOC)

    def __init__(
        self,
        *,
        pool: ConnectionPool,
        local_toolboxes: Optional[List[LocalToolbox]] = None,
        tool_map: Optional[Dict[str, Any]] = None,
        refresh_interval: float = 0.0,
        logger: Optional[logging.Logger] = None,
        enable_meta_tools: bool | Iterable[str] = True,
    ) -> None:
        self._pool = pool
        self._local_boxes: List[LocalToolbox] = list(local_toolboxes or [])
        self._tool_map: Dict[str, Any] = tool_map if tool_map is not None else {}

        self._refresh_interval: float = float(refresh_interval)
        self._logger = logger or logging.getLogger("seqoria_agent.tool_registry")

        # Meta-tool toggle --------------------------------------------------
        self._enabled_meta_names: Set[str]

        # Normalise enable_meta_tools into a set of internal names
        if enable_meta_tools is True:
            self._enabled_meta_names = {THINK_TOOL_NAME, CLARIFY_TOOL_NAME, SUMMARIZE_TOOL_NAME}
        elif enable_meta_tools is False:
            self._enabled_meta_names = set()
        else:
            # Accept any iterable of string-like names (case-insensitive)
            mapping = {
                "think": THINK_TOOL_NAME,
                "__think__": THINK_TOOL_NAME,
                "clarify": CLARIFY_TOOL_NAME,
                "__clarify__": CLARIFY_TOOL_NAME,
                "summarize": SUMMARIZE_TOOL_NAME,
                "__summarize__": SUMMARIZE_TOOL_NAME,
            }
            chosen: Set[str] = set()
            for n in enable_meta_tools:  # type: ignore[arg-type]
                try:
                    canon = mapping.get(str(n).lower())
                    if canon:
                        chosen.add(canon)
                except Exception:
                    continue
            self._enabled_meta_names = chosen

        # Internal state ---------------------------------------------------
        self._llm_tools: List[dict] = []
        self._server_params: Dict[str, List[str]] = {}
        self._display_tools: Set[str] = set()
        self._tool_deps: Dict[str, Set[str]] = {}

        # Concurrency guard for refreshes
        self._lock = asyncio.Lock()

        # Background refresher task
        self._refresher_task: Optional[asyncio.Task] = None
        if self._refresh_interval > 0:
            loop = asyncio.get_running_loop()
            self._refresher_task = loop.create_task(self._periodic_refresh())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def llm_tools(self) -> List[dict]:  # noqa: D401 – simple getter
        """Return OpenAI-compatible tool list for the LLM interface."""
        return list(self._llm_tools)

    def is_display_tool(self, name: str) -> bool:  # noqa: D401
        """Return *True* when *name* is annotated as a display-tool."""
        return name in self._display_tools

    def server_params(self, name: str) -> List[str]:  # noqa: D401
        """Return server-populated parameter names for *name* (may be empty)."""
        return list(self._server_params.get(name, []))

    def depends_on(self, name: str) -> Set[str]:  # noqa: D401
        """Return dependency set for *name* (tools that must run first)."""
        return set(self._tool_deps.get(name, set()))

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: D401 – simple coro
        """Stop background task (if any)."""
        if self._refresher_task and not self._refresher_task.done():
            self._refresher_task.cancel()
            try:
                await self._refresher_task
            except Exception:  # pragma: no cover – ignore
                pass

    # ------------------------------------------------------------------
    # Main refresh implementation (extracted from AIClient._update_llm_tools)
    # ------------------------------------------------------------------

    async def refresh(self) -> None:  # noqa: D401 – primary public coroutine
        """(Re)build the tool catalogue from remote + local sources."""

        async with self._lock:
            # Clear previous state – callers may hold on to llm_tools list, so
            # we create *new* containers below and re-assign atomically.
            self._tool_map.clear()
            self._display_tools.clear()
            self._server_params.clear()
            self._tool_deps.clear()

        llm_tools: List[dict] = []

        # ------------------------------------------------------------------
        # 1) Merge REMOTE tools
        # ------------------------------------------------------------------
        connections = self._pool.connections
        if connections:
            tool_results = await asyncio.gather(
                *(conn.client.list_tools() for conn in connections),
                return_exceptions=True,
            )
            for idx, result in enumerate(tool_results):
                conn = connections[idx]
                if isinstance(result, Exception):
                    self._logger.warning(
                        "Could not list tools from server", url=conn.url, error=str(result)
                    )
                    continue

                for tool in result:
                    if tool.name in self._tool_map:
                        # Remote-remote duplicate (different servers) – prefer first seen
                        continue

                    self._tool_map[tool.name] = conn.client

                    # -------- Annotation parsing (supports dict & pydantic) ---------
                    annotations = getattr(tool, "annotations", None)
                    if annotations:
                        # Legacy dict-style
                        if isinstance(annotations, dict):
                            if annotations.get("is_display_tool"):
                                self._display_tools.add(tool.name)
                            sp = annotations.get("server_populated") or []
                            if isinstance(sp, list):
                                self._server_params[tool.name] = list(sp)
                            dep = annotations.get("depends_on") or []
                            if isinstance(dep, list):
                                self._tool_deps[tool.name] = set(dep)
                        else:
                            # Likely a *pydantic* model – try attribute access with fallbacks
                            is_display = False
                            if hasattr(annotations, "is_display_tool"):
                                is_display = bool(annotations.is_display_tool)
                            elif hasattr(annotations, "model_dump"):
                                try:
                                    is_display = bool(annotations.model_dump().get("is_display_tool"))
                                except Exception:
                                    pass
                            if is_display:
                                self._display_tools.add(tool.name)

                            def _safe_get(field: str):
                                if hasattr(annotations, field):
                                    return getattr(annotations, field)
                                if hasattr(annotations, "model_dump"):
                                    try:
                                        return annotations.model_dump().get(field)
                                    except Exception:
                                        return None
                                return None

                            sp_list = _safe_get("server_populated")
                            if sp_list:
                                self._server_params[tool.name] = list(sp_list)

                            dep_list = _safe_get("depends_on")
                            if dep_list:
                                self._tool_deps[tool.name] = set(dep_list)

                    llm_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,  # noqa: N803 external name
                            },
                        }
                    )

        # ------------------------------------------------------------------
        # 2) Merge LOCAL toolbox tools
        # ------------------------------------------------------------------
        for box in self._local_boxes:
            for name_, info in box._tools.items():  # pylint: disable=protected-access
                if name_ in self._tool_map:
                    # Duplicate between remote and local – raise as this is a config error
                    raise RuntimeError(
                        f"Duplicate tool name '{name_}' between remote server and local toolbox"
                    )
                self._tool_map[name_] = box  # type: ignore[assignment]

                if info.display:
                    self._display_tools.add(name_)
                if info.server_params:
                    self._server_params[name_] = list(info.server_params)
                if info.depends_on:
                    self._tool_deps[name_] = set(info.depends_on)

                llm_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name_,
                            "description": info.description,
                            "parameters": info.input_schema,
                        },
                    }
                )

        # ------------------------------------------------------------------
        # 3)  Built-in meta-tools (optional)
        # ------------------------------------------------------------------

        if self._enabled_meta_names:
            llm_tools.extend([t for t in _META_TOOLS if t["function"]["name"] in self._enabled_meta_names])

        # ------------------------------------------------------------------
        # Finalise – replace list atomically so concurrent reads see either old or new
        # ------------------------------------------------------------------
        async with self._lock:
            self._llm_tools = llm_tools

        self._logger.info("ToolRegistry refreshed", tool_count=len(llm_tools))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _periodic_refresh(self) -> None:
        """Background task that refreshes the catalogue periodically."""
        try:
            while True:
                await asyncio.sleep(self._refresh_interval)
                try:
                    await self.refresh()
                except Exception as exc:  # pragma: no cover – keep alive
                    self._logger.warning("Background tool refresh failed", error=str(exc))
        except asyncio.CancelledError:
            # Normal shutdown path
            return 

# ---------------------------------------------------------------------------
# Built-in meta-tools (OpenAI tool schema) – keep at bottom to avoid circulars
# ---------------------------------------------------------------------------

_META_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": THINK_TOOL_NAME,
            "description": (
                "Internal helper - allows the assistant to think out loud "
                "without exposing the thought to the user."),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "The assistant's internal reasoning.",
                    },
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CLARIFY_TOOL_NAME,
            "description": (
                "Ask the user a concise question when additional information "
                "is required."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarification question for the user.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SUMMARIZE_TOOL_NAME,
            "description": (
                "Compresses the current chat history so the conversation can "
                "continue within context limits."),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
] 