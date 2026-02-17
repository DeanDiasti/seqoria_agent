from __future__ import annotations

from typing import Any, Literal, Protocol

from seqoria_agent.memory.models import Checkpoint, MemoryEdge, MemoryNode, MemoryScope, Plan


class VectorStoreProtocol(Protocol):
    """Vector index protocol used by MemoryGraph."""

    def upsert(self, item_id: str, embedding: list[float], *, scope: MemoryScope) -> None: ...
    def remove(self, item_id: str, *, scope: MemoryScope) -> None: ...
    def search(
        self,
        embedding: list[float],
        *,
        scope: MemoryScope,
        top_k: int,
    ) -> list[tuple[str, float]]: ...
    def delete(self, node_id: str, *, scope: MemoryScope) -> None: ...


class MemoryRepositoryProtocol(Protocol):
    """Canonical persistence protocol for memory records."""

    # Nodes -----------------------------------------------------------------
    def upsert_node(self, *, scope: MemoryScope, node: MemoryNode) -> None: ...
    def get_node(self, *, scope: MemoryScope, node_id: str) -> MemoryNode | None: ...
    def list_nodes(
        self,
        *,
        scope: MemoryScope,
        memory_type: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]: ...
    def remove_node(self, *, scope: MemoryScope, node_id: str) -> None: ...
    def set_node_index_state(
        self,
        *,
        scope: MemoryScope,
        node_id: str,
        status: Literal["pending", "indexed", "failed"],
        index_version: int | None = None,
    ) -> None: ...

    # Edges -----------------------------------------------------------------
    def upsert_edge(self, *, scope: MemoryScope, edge: MemoryEdge) -> None: ...
    def list_edges(self, *, scope: MemoryScope, node_id: str | None = None) -> list[MemoryEdge]: ...

    # Checkpoints -----------------------------------------------------------
    def save_checkpoint(self, *, scope: MemoryScope, checkpoint: Checkpoint) -> None: ...
    def list_checkpoints(self, *, scope: MemoryScope) -> list[Checkpoint]: ...

    # Timeline / events -----------------------------------------------------
    def append_event(self, *, scope: MemoryScope, event: dict[str, Any]) -> None: ...
    def list_events(self, *, scope: MemoryScope, limit: int | None = None) -> list[dict[str, Any]]: ...

    # Plan ------------------------------------------------------------------
    def save_plan(self, *, scope: MemoryScope, plan: Plan | None) -> None: ...
    def get_plan(self, *, scope: MemoryScope) -> Plan | None: ...


class PersistenceStoreProtocol(Protocol):
    """Scope-aware persistence store for AgenticMemory state."""

    async def save_state(
        self,
        *,
        scope: MemoryScope,
        state: dict[str, Any],
        path: str | None = None,
    ) -> str: ...

    async def load_state(
        self,
        *,
        scope: MemoryScope,
        path: str | None = None,
    ) -> dict[str, Any] | None: ...

