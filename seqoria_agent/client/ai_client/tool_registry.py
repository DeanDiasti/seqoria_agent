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
import hashlib
import json
import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Set
from functools import lru_cache

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.ai_client.constants import THINK_TOOL_NAME

__all__: list[str] = ["ToolRegistry"]

# Phase-2 optimisation ------------------------------------------------------
# Use xxhash when available and cache fingerprint computations.

try:
    import xxhash  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    xxhash = None


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

    def __init__(self,
                 *,
                 pool: ConnectionPool,
                 local_toolboxes: Optional[List[LocalToolbox]] = None,
                 tool_map: Optional[Dict[str, Any]] = None,
                 refresh_interval: float = 0.0,
                 logger: Optional[logging.Logger] = None,
                 enable_meta_tools: bool | Iterable[str] = True) -> None:
        self._pool = pool
        self._local_boxes: List[LocalToolbox] = list(local_toolboxes or [])
        self._tool_map: Dict[str, Any] = tool_map if tool_map is not None else {}

        self._refresh_interval: float = float(refresh_interval)
        self._logger = logger or logging.getLogger(name="seqoria_agent.tool_registry")

        # Meta-tool toggle --------------------------------------------------
        self._enabled_meta_names: Set[str]

        self._enabled_meta_names = {THINK_TOOL_NAME} if enable_meta_tools else set()

        # Internal state ---------------------------------------------------
        self._llm_tools: List[dict] = []
        self._server_params: Dict[str, List[str]] = {}
        self._tool_deps: Dict[str, Set[str]] = {}
        self._seen_schema_hash: Dict[str, str] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._refresher_task: Optional[asyncio.Task] = None
        # Version counter used by Executor cache (increments every successful refresh)
        self._version: int = 0
        if self._refresh_interval > 0:
            self._refresher_task = asyncio.get_running_loop().create_task(coro=self._periodic_refresh())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def llm_tools(self) -> List[dict]:  # noqa: D401 – simple getter
        """Return OpenAI-compatible tool list for the LLM interface."""
        return list(self._llm_tools)

    def server_params(self, name: str) -> List[str]:  # noqa: D401
        """Return server-populated parameter names for *name* (may be empty)."""
        return list(self._server_params.get(name, []))

    def depends_on(self, name: str) -> Set[str]:  # noqa: D401
        """Return dependency set for *name* (tools that must run first)."""
        return set(self._tool_deps.get(name, set()))

    @property
    def version(self) -> int:  # noqa: D401
        """Monotonically increasing version number bumped on every refresh."""
        return self._version

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
            self._server_params.clear()
            self._tool_deps.clear()
            self._seen_schema_hash.clear()

        llm_tools: List[dict] = []

        # A mapping from *base* tool name → variant alias we produced for the FIRST
        # variant (used so subsequent tools from the same server can refer to it
        # in their `depends_on` annotations).
        base_to_primary_alias: Dict[str, str] = {}

        # Merge REMOTE tools
        connections = self._pool.connections
        if connections:
            tool_results = await asyncio.gather(*(conn.client.list_tools() for conn in connections),
                                                return_exceptions=True)
            for idx, result in enumerate(iterable=tool_results):
                conn = connections[idx]
                if isinstance(result, Exception):
                    self._logger.warning("Could not list tools from server (%s): %s", conn.url, result)
                    continue

                for tool in result:
                    base_name: str = tool.name
                    schema_fingerprint = self._schema_hash(schema=tool.inputSchema)

                    # Decide public name ------------------------------------------------
                    primary_hash = self._seen_schema_hash.get(base_name)
                    if primary_hash is None:
                        # First time we see this name → keep the short variant
                        public_name = base_name
                        self._seen_schema_hash[base_name] = schema_fingerprint
                        base_to_primary_alias[base_name] = public_name
                    elif primary_hash == schema_fingerprint:
                        # Same schema – duplicate registration (replica). Skip additional
                        # copies for now but we *could* aggregate back-ends later.
                        continue
                    else:
                        # New schema variant → create alias with fingerprint suffix
                        public_name = f"{base_name}__{schema_fingerprint}"

                    # ------------------------------------------------------------------
                    # Register mapping + annotation-derived metadata
                    self._tool_map[public_name] = conn.client

                    # Annotation parsing (supports dict & pydantic) ---------
                    annotations = getattr(tool, "annotations", None)  # type: ignore[attr-defined]
                    if annotations:
                        # Extract fields once using helper
                        sp_raw = self._get_annotation_field(annotations=annotations, 
                                                            field="server_populated")
                        if sp_raw:
                            self._server_params[public_name] = list(sp_raw)

                        dep_raw = self._get_annotation_field(annotations=annotations, 
                                                             field="depends_on")
                        if dep_raw:
                            mapped = {base_to_primary_alias.get(d, d) for d in dep_raw}
                            self._tool_deps[public_name] = mapped

                    llm_tools.append({"type": "function",
                                      "function": {"name": public_name,
                                                   "description": tool.description,
                                                   "parameters": tool.inputSchema}})  # noqa: N803 external name

        # ------------------------------------------------------------------
        # 2) Merge LOCAL toolbox tools
        # ------------------------------------------------------------------
        for box in self._local_boxes:
            for base_name, info in box._tools.items():
                schema_fingerprint = self._schema_hash(schema=info.input_schema)

                primary_hash = self._seen_schema_hash.get(base_name)
                if primary_hash is None:
                    public_name = base_name
                    self._seen_schema_hash[base_name] = schema_fingerprint
                    base_to_primary_alias[base_name] = public_name
                elif primary_hash == schema_fingerprint:
                    # Same schema already registered (remote replica) -> skip registering again
                    continue
                else:
                    public_name = f"{base_name}__{schema_fingerprint}"

                self._tool_map[public_name] = box  # type: ignore[assignment]

                if info.server_params:
                    self._server_params[public_name] = list(info.server_params)
                if info.depends_on:
                    self._tool_deps[public_name] = {base_to_primary_alias.get(d, d) for d in info.depends_on}

                llm_tools.append({"type": "function",
                                  "function": {"name": public_name,
                                               "description": info.description,
                                               "parameters": info.input_schema}})

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

            # Bump version after successful rebuild so dependants can invalidate caches
            self._version += 1

        self._logger.info("ToolRegistry refreshed with %d tools", len(llm_tools))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _periodic_refresh(self) -> None:
        """Background task that refreshes the catalogue periodically."""
        try:
            while True:
                await asyncio.sleep(delay=self._refresh_interval)
                try:
                    await self.refresh()
                except Exception as exc:  # pragma: no cover – keep alive
                    self._logger.warning(msg="Background tool refresh failed", 
                                         error=str(object=exc))
        except asyncio.CancelledError:
            # Normal shutdown path
            return 

    # ------------------------------------------------------------------
    # Internal hashing helpers (Phase-2 optimisation)
    # ------------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=4096)
    def _fingerprint(canonical_json: str) -> str:  # noqa: D401
        """Return an 8-char stable fingerprint for *canonical_json*."""
        if xxhash is not None:
            # xxhash expects bytes; encode once for efficiency
            return xxhash.xxh3_64_hexdigest(canonical_json.encode())[:8]
        # Fallback to cryptographic hash when xxhash unavailable
        return hashlib.sha1(canonical_json.encode()).hexdigest()[:8]

    @staticmethod
    def _schema_hash(schema: dict[str, Any]) -> str:  # noqa: D401
        """Return an 8-char fingerprint for *schema* (deterministic, cached)."""
        canonical_json = json.dumps(obj=schema, sort_keys=True, separators=(",", ":"))
        return ToolRegistry._fingerprint(canonical_json)

    @staticmethod
    def _get_annotation_field(annotations: Any, field: str):  # noqa: ANN401
        """Return *field* value from *annotations* supporting dict / pydantic."""
        if annotations is None:
            return None
        # Plain dict (legacy)
        if isinstance(annotations, dict):
            return annotations.get(field)

        # Attribute access (pydantic or SimpleNamespace)
        if hasattr(annotations, field):
            return getattr(annotations, field)

        # Pydantic v2 – use model_dump when available
        if hasattr(annotations, "model_dump"):
            try:
                return annotations.model_dump().get(field)
            except Exception:
                return None

        return None

_META_TOOLS: list[dict] = [{"type": "function",
                            "function": {"name": THINK_TOOL_NAME,
                                         "description": ("Internal helper - allows the assistant to think out loud "
                                                         "without exposing the thought to the user."),
                                         "parameters": {"type": "object",
                                                        "properties": {"thought": {"type": "string",
                                                                                   "description": "The assistant's internal reasoning."}},
                                                        "required": ["thought"]}}}] 