from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Literal

from seqoria_agent.memory.models import Checkpoint, MemoryEdge, MemoryNode, MemoryScope, Plan
from seqoria_agent.memory.stores.base import MemoryRepositoryProtocol, VectorStoreProtocol


class InMemoryVectorStore(VectorStoreProtocol):
    """Simple scope-aware in-memory vector index."""

    def __init__(self) -> None:
        self._vectors: dict[str, dict[str, list[float]]] = defaultdict(dict)

    def upsert(self, item_id: str, embedding: list[float], *, scope: MemoryScope) -> None:
        self._vectors[scope.key()][item_id] = embedding

    def remove(self, item_id: str, *, scope: MemoryScope) -> None:
        scope_key = scope.key()
        self._vectors.get(scope_key, {}).pop(item_id, None)

    def search(
        self,
        embedding: list[float],
        *,
        scope: MemoryScope,
        top_k: int,
    ) -> list[tuple[str, float]]:
        scope_vectors = self._vectors.get(scope.key(), {})
        scored: list[tuple[str, float]] = []
        for item_id, candidate in scope_vectors.items():
            scored.append((item_id, self._cosine_similarity(embedding, candidate)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def delete(self, node_id: str, *, scope: MemoryScope) -> None:
        self._vectors_by_scope[scope.key()].pop(node_id, None)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(a * a for a in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)


class InMemoryMemoryRepository(MemoryRepositoryProtocol):
    """Canonical in-memory repository used for tests/dev."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, MemoryNode]] = defaultdict(dict)
        self._edges: dict[str, list[MemoryEdge]] = defaultdict(list)
        self._checkpoints: dict[str, list[Checkpoint]] = defaultdict(list)
        self._events: dict[str, list[dict]] = defaultdict(list)
        self._plans: dict[str, Plan | None] = {}

    def _scope_key(self, scope: MemoryScope) -> str:
        return scope.key()

    # Nodes -----------------------------------------------------------------
    def upsert_node(self, *, scope: MemoryScope, node: MemoryNode) -> None:
        self._nodes[self._scope_key(scope)][node.id] = copy.deepcopy(node)

    def get_node(self, *, scope: MemoryScope, node_id: str) -> MemoryNode | None:
        node = self._nodes[self._scope_key(scope)].get(node_id)
        return copy.deepcopy(node) if node else None

    def list_nodes(
        self,
        *,
        scope: MemoryScope,
        memory_type: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        items = list(self._nodes[self._scope_key(scope)].values())
        if memory_type is not None:
            items = [item for item in items if item.memory_type == memory_type]
        items.sort(key=lambda node: node.created_at)
        if limit is not None:
            items = items[:limit]
        return [copy.deepcopy(item) for item in items]

    def remove_node(self, *, scope: MemoryScope, node_id: str) -> None:
        self._nodes[self._scope_key(scope)].pop(node_id, None)
        # Keep edges list compact by removing references.
        self._edges[self._scope_key(scope)] = [
            edge
            for edge in self._edges[self._scope_key(scope)]
            if edge.source_id != node_id and edge.target_id != node_id
        ]

    def set_node_index_state(
        self,
        *,
        scope: MemoryScope,
        node_id: str,
        status: Literal["pending", "indexed", "failed"],
        index_version: int | None = None,
    ) -> None:
        node = self._nodes[self._scope_key(scope)].get(node_id)
        if node is None:
            return
        node.index_status = status
        if index_version is not None:
            node.index_version = index_version
        self._nodes[self._scope_key(scope)][node_id] = node

    # Edges -----------------------------------------------------------------
    def upsert_edge(self, *, scope: MemoryScope, edge: MemoryEdge) -> None:
        scope_key = self._scope_key(scope)
        for idx, existing in enumerate(self._edges[scope_key]):
            if (
                existing.source_id == edge.source_id
                and existing.target_id == edge.target_id
                and existing.relationship == edge.relationship
            ):
                self._edges[scope_key][idx] = copy.deepcopy(edge)
                return
        self._edges[scope_key].append(copy.deepcopy(edge))

    def list_edges(self, *, scope: MemoryScope, node_id: str | None = None) -> list[MemoryEdge]:
        edges = self._edges[self._scope_key(scope)]
        if node_id is None:
            return [copy.deepcopy(edge) for edge in edges]
        filtered = [
            edge
            for edge in edges
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        return [copy.deepcopy(edge) for edge in filtered]

    # Checkpoints -----------------------------------------------------------
    def save_checkpoint(self, *, scope: MemoryScope, checkpoint: Checkpoint) -> None:
        self._checkpoints[self._scope_key(scope)].append(copy.deepcopy(checkpoint))

    def list_checkpoints(self, *, scope: MemoryScope) -> list[Checkpoint]:
        return [copy.deepcopy(cp) for cp in self._checkpoints[self._scope_key(scope)]]

    # Timeline / events -----------------------------------------------------
    def append_event(self, *, scope: MemoryScope, event: dict) -> None:
        self._events[self._scope_key(scope)].append(copy.deepcopy(event))

    def list_events(self, *, scope: MemoryScope, limit: int | None = None) -> list[dict]:
        items = self._events[self._scope_key(scope)]
        if limit is not None:
            items = items[-limit:]
        return [copy.deepcopy(event) for event in items]

    # Plan ------------------------------------------------------------------
    def save_plan(self, *, scope: MemoryScope, plan: Plan | None) -> None:
        self._plans[self._scope_key(scope)] = copy.deepcopy(plan)

    def get_plan(self, *, scope: MemoryScope) -> Plan | None:
        plan = self._plans.get(self._scope_key(scope))
        return copy.deepcopy(plan) if plan else None

