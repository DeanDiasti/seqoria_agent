"""Tool Registry for discovering and cataloging MCP tools.

This module manages tool discovery from remote MCP servers and local toolboxes,
maintaining a unified catalogue for LLM interaction.

Features:
- Multi-source tool discovery (MCP servers + LocalToolbox)
- Schema deduplication via fingerprinting
- Server-populated parameter tracking
- Tool dependency tracking
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Set, FrozenSet

from mcp import ClientSession

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.ai_client.utils import json_dumps, fast_hash
from seqoria_agent.toolbox import LocalToolbox

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """Central catalogue of tool metadata from MCP servers and local toolboxes.

    Parameters
    ----------
    pool : MCPConnectionPool
        Connection pool for remote MCP servers.
    local_toolboxes : list[LocalToolbox] | None
        Local toolboxes to include in the catalogue.
    tool_map : dict | None
        Mutable mapping of tool_name -> session/toolbox (shared with pool).
    refresh_interval : float
        Interval for automatic background refresh (0 = disabled).
    logger : logging.Logger | None
        Logger for registry events.
    """

    def __init__(self,
                 *,
                 pool: MCPConnectionPool,
                 local_toolboxes: Optional[List[LocalToolbox]] = None,
                 tool_map: Optional[Dict[str, Any]] = None,
                 blocked_tools: Iterable[str] | None = None,
                 refresh_interval: float = 0.0,
                 logger: Optional[logging.Logger] = None) -> None:
        self._pool = pool
        self._local_boxes: List[LocalToolbox] = list(local_toolboxes or [])
        self._tool_map: Dict[str, Any] = tool_map if tool_map is not None else {}
        self._blocked_tools: FrozenSet[str] = frozenset(str(n) for n in (blocked_tools or []))

        self._refresh_interval: float = float(refresh_interval)
        self._logger = logger or logging.getLogger(name="seqoria_agent.tool_registry")

        # Internal state
        self._llm_tools: List[dict] = []
        # Store immutable containers internally to avoid accidental mutation and
        # to make cheap "view" access possible for hot paths.
        self._server_params: Dict[str, tuple[str, ...]] = {}
        self._tool_deps: Dict[str, frozenset[str]] = {}
        self._seen_schema_hash: Dict[str, str] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._refresher_task: Optional[asyncio.Task] = None
        self._version: int = 0

        # Start background refresh if configured
        if self._refresh_interval > 0:
            try:
                loop = asyncio.get_running_loop()
                self._refresher_task = loop.create_task(self._periodic_refresh())
            except RuntimeError:
                pass  # No running loop yet

    @property
    def llm_tools(self) -> List[dict]:
        """Return OpenAI-compatible tool list for LLM interface."""
        return list(self._llm_tools)

    @property
    def blocked_tools(self) -> Set[str]:
        """Return the currently blocked tool names (copy)."""
        return set(self._blocked_tools)

    def _llm_tools_view(self) -> List[dict]:
        """Internal: return the underlying tool list without copying (read-only contract)."""
        return self._llm_tools

    def server_params(self, name: str) -> List[str]:
        """Return server-populated parameter names for a tool."""
        return list(self._server_params.get(name, ()))

    def _server_params_view(self, name: str) -> tuple[str, ...]:
        """Internal: return server-populated parameter names without allocating."""
        return self._server_params.get(name, ())

    def depends_on(self, name: str) -> Set[str]:
        """Return dependency set for a tool (tools that must run first)."""
        return set(self._tool_deps.get(name, frozenset()))

    def _depends_on_view(self, name: str) -> frozenset[str]:
        """Internal: return dependencies without allocating."""
        return self._tool_deps.get(name, frozenset())

    @property
    def version(self) -> int:
        """Monotonically increasing version bumped on each refresh."""
        return self._version

    async def set_blocked_tools(self, names: Iterable[str]) -> None:
        """Replace the blocked-tool set.

        Callers typically follow this with `await registry.refresh()` so the tool
        catalogue reflects the new filter.
        """
        async with self._lock:
            self._blocked_tools = frozenset(str(n) for n in names)

    async def block_tools(self, names: Iterable[str]) -> None:
        """Add tool names to the blocklist."""
        async with self._lock:
            self._blocked_tools = frozenset(set(self._blocked_tools).union({str(n) for n in names}))

    async def unblock_tools(self, names: Iterable[str]) -> None:
        """Remove tool names from the blocklist."""
        to_remove = {str(n) for n in names}
        async with self._lock:
            self._blocked_tools = frozenset(n for n in self._blocked_tools if n not in to_remove)

    def _is_blocked(self, *, base_name: str, public_name: str) -> bool:
        """Return True if a tool should be excluded from the LLM tool context."""
        blocked = self._blocked_tools
        # - base name block should hide all schema aliases (`foo__abcd1234`)
        # - public name block can hide a single schema variant
        if not blocked:
            return False
        if base_name in blocked or public_name in blocked:
            return True
        if public_name.startswith(base_name + "__") and base_name in blocked:
            return True
        return False

    async def close(self) -> None:
        """Stop background refresh task."""
        if self._refresher_task and not self._refresher_task.done():
            self._refresher_task.cancel()
            try:
                await self._refresher_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def refresh(self) -> None:
        """Rebuild tool catalogue from all sources."""
        async with self._lock:
            self._tool_map.clear()
            self._server_params.clear()
            self._tool_deps.clear()
            self._seen_schema_hash.clear()

        llm_tools: List[dict] = []
        base_to_primary_alias: Dict[str, str] = {}

        # 1) Discover tools from remote MCP servers
        await self._refresh_remote_tools(llm_tools, base_to_primary_alias)

        # 2) Merge local toolbox tools
        await self._refresh_local_tools(llm_tools, base_to_primary_alias)

        # Atomically update state
        async with self._lock:
            self._llm_tools = llm_tools
            self._version += 1

        self._logger.info("ToolRegistry refreshed with %d tools", len(llm_tools))

    async def _refresh_remote_tools(self,
                                    llm_tools: List[dict],
                                    base_to_primary_alias: Dict[str, str]) -> None:
        """Discover tools from remote MCP servers."""
        connections = self._pool.connections
        if not connections:
            return

        # Gather tool lists from all connections
        tasks = []
        for conn in connections:
            tasks.append(self._list_tools_safe(conn.session, conn.url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            conn = connections[idx]

            if isinstance(result, Exception):
                self._logger.warning("Could not list tools from server (%s): %s", conn.url, result)
                continue

            if not result:
                continue

            for tool in result:
                self._register_tool(
                    tool=tool,
                    source=conn.session,
                    llm_tools=llm_tools,
                    base_to_primary_alias=base_to_primary_alias,
                )

    async def _list_tools_safe(self, session: ClientSession, url: str) -> List[Any]:
        """Safely list tools from an MCP session."""
        try:
            result = await session.list_tools()
            return list(result.tools) if hasattr(result, "tools") else []
        except Exception as exc:
            self._logger.warning("Failed to list tools from %s: %s", url, exc)
            return []

    async def _refresh_local_tools(self, llm_tools: List[dict], base_to_primary_alias: Dict[str, str]) -> None:
        """Merge tools from local toolboxes."""
        for box in self._local_boxes:
            for tool_info in box.list_tools():
                self._register_local_tool(
                    tool_info=tool_info,
                    source=box,
                    llm_tools=llm_tools,
                    base_to_primary_alias=base_to_primary_alias,
                )

    def _register_tool(
        self,
        tool: Any,
        source: ClientSession,
        llm_tools: List[dict],
        base_to_primary_alias: Dict[str, str],
    ) -> None:
        """Register a tool from a remote MCP server."""
        base_name: str = tool.name
        
        # Get input schema
        input_schema = {}
        if hasattr(tool, "inputSchema"):
            input_schema = tool.inputSchema
        elif hasattr(tool, "input_schema"):
            input_schema = tool.input_schema
        
        schema_fingerprint = self._schema_hash(input_schema)

        # Determine public name (handle duplicates)
        primary_hash = self._seen_schema_hash.get(base_name)
        if primary_hash is None:
            public_name = base_name
        elif primary_hash == schema_fingerprint:
            # Same schema - skip duplicate
            return
        else:
            # Different schema - create alias
            public_name = f"{base_name}__{schema_fingerprint}"

        # Apply blocklist before mutating registry state.
        if self._is_blocked(base_name=base_name, public_name=public_name):
            return

        # Now commit primary mapping once we know the tool isn't blocked.
        if primary_hash is None:
            self._seen_schema_hash[base_name] = schema_fingerprint
            base_to_primary_alias[base_name] = public_name

        # Register in tool map
        self._tool_map[public_name] = source

        # Parse annotations
        annotations = getattr(tool, "annotations", None)
        if annotations:
            sp_raw = self._get_annotation_field(annotations, "server_populated")
            if sp_raw:
                self._server_params[public_name] = tuple(sp_raw)

            dep_raw = self._get_annotation_field(annotations, "depends_on")
            if dep_raw:
                mapped = {base_to_primary_alias.get(d, d) for d in dep_raw}
                self._tool_deps[public_name] = frozenset(mapped)

        # Add to LLM tools list
        description = getattr(tool, "description", "") or ""
        llm_tools.append({
            "type": "function",
            "function": {
                "name": public_name,
                "description": description,
                "parameters": input_schema,
            },
        })

    def _register_local_tool(
        self,
        tool_info: Any,
        source: LocalToolbox,
        llm_tools: List[dict],
        base_to_primary_alias: Dict[str, str],
    ) -> None:
        """Register a tool from a local toolbox."""
        base_name: str = tool_info.name
        input_schema = getattr(tool_info, "inputSchema", {})
        schema_fingerprint = self._schema_hash(input_schema)

        # Determine public name
        primary_hash = self._seen_schema_hash.get(base_name)
        if primary_hash is None:
            public_name = base_name
        elif primary_hash == schema_fingerprint:
            return
        else:
            public_name = f"{base_name}__{schema_fingerprint}"

        # Apply blocklist before mutating registry state.
        if self._is_blocked(base_name=base_name, public_name=public_name):
            return

        # Commit primary mapping once we know the tool isn't blocked.
        if primary_hash is None:
            self._seen_schema_hash[base_name] = schema_fingerprint
            base_to_primary_alias[base_name] = public_name

        # Register in tool map
        self._tool_map[public_name] = source

        # Parse annotations
        annotations = getattr(tool_info, "annotations", None)
        if annotations:
            sp_raw = self._get_annotation_field(annotations, "server_populated")
            if sp_raw:
                self._server_params[public_name] = tuple(sp_raw)

            dep_raw = self._get_annotation_field(annotations, "depends_on")
            if dep_raw:
                mapped = {base_to_primary_alias.get(d, d) for d in dep_raw}
                self._tool_deps[public_name] = frozenset(mapped)

        # Add to LLM tools list
        description = getattr(tool_info, "description", "") or ""
        llm_tools.append({
            "type": "function",
            "function": {
                "name": public_name,
                "description": description,
                "parameters": input_schema,
            },
        })

    async def _periodic_refresh(self) -> None:
        """Background task for periodic refresh."""
        try:
            while True:
                await asyncio.sleep(self._refresh_interval)
                try:
                    await self.refresh()
                except Exception as exc:
                    self._logger.warning("Background refresh failed: %s", exc)
        except asyncio.CancelledError:
            return

    @staticmethod
    def _schema_hash(schema: dict[str, Any]) -> str:
        """Return 8-char fingerprint for a schema dict."""
        canonical = json_dumps(schema, sort_keys=True)
        return fast_hash(canonical)[:8]

    @staticmethod
    def _get_annotation_field(annotations: Any, field: str) -> Any:
        """Extract field from annotations (dict or object)."""
        if annotations is None:
            return None

        if isinstance(annotations, dict):
            return annotations.get(field)

        if hasattr(annotations, field):
            return getattr(annotations, field)

        if hasattr(annotations, "model_dump"):
            try:
                return annotations.model_dump().get(field)
            except Exception:
                return None

        return None


