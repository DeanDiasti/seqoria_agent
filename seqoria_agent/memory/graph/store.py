from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Literal

from seqoria_agent.memory.embeddings.base import EmbedderProtocol
from seqoria_agent.memory.graph.node import node_matches_scope, node_text_for_retrieval
from seqoria_agent.memory.models import MemoryEdge, MemoryNode, MemoryScope
from seqoria_agent.memory.stores.base import MemoryRepositoryProtocol, VectorStoreProtocol
from seqoria_agent.memory.stores.in_memory import InMemoryMemoryRepository, InMemoryVectorStore


class MemoryGraph:
    """Scope-aware memory graph with repo hydration + vector retrieval."""

    def __init__(
        self,
        *,
        vector_store: VectorStoreProtocol | None = None,
        repository: MemoryRepositoryProtocol | None = None,
    ) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: list[MemoryEdge] = []
        self._adjacency: dict[str, list[MemoryEdge]] = defaultdict(list)
        self._vector_store = vector_store or InMemoryVectorStore()
        self._repository = repository or InMemoryMemoryRepository()

    @property
    def nodes(self) -> dict[str, MemoryNode]:
        return self._nodes

    @property
    def edges(self) -> list[MemoryEdge]:
        return self._edges

    def add_node(self, node: MemoryNode) -> None:
        self._repository.upsert_node(scope=node.scope, node=node)
        self._nodes[node.id] = node
        if node.embedding:
            self._vector_store.upsert(node.id, node.embedding, scope=node.scope)

    def add_edge(self, edge: MemoryEdge, *, scope: MemoryScope) -> None:
        self._repository.upsert_edge(scope=scope, edge=edge)
        self._edges.append(edge)
        self._adjacency[edge.source_id].append(edge)

    def get_node(self, node_id: str, *, scope: MemoryScope | None = None) -> MemoryNode | None:
        cached = self._nodes.get(node_id)
        if cached is not None:
            if scope is None or cached.scope.matches(scope):
                return cached
        if scope is None:
            return cached
        hydrated = self._repository.get_node(scope=scope, node_id=node_id)
        if hydrated:
            self._nodes[hydrated.id] = hydrated
        return hydrated

    def list_nodes(self, *, scope: MemoryScope | None = None) -> list[MemoryNode]:
        if scope is None:
            return list(self._nodes.values())
        hydrated = self._repository.list_nodes(scope=scope)
        for node in hydrated:
            self._nodes[node.id] = node
        return [item for item in hydrated if node_matches_scope(item, scope)]

    def list_edges(
        self,
        *,
        scope: MemoryScope | None = None,
        node_id: str | None = None,
    ) -> list[MemoryEdge]:
        if scope is None:
            if node_id is None:
                return list(self._edges)
            return [
                edge
                for edge in self._edges
                if edge.source_id == node_id or edge.target_id == node_id
            ]
        hydrated = self._repository.list_edges(scope=scope, node_id=node_id)
        for edge in hydrated:
            exists = any(
                existing.source_id == edge.source_id
                and existing.target_id == edge.target_id
                and existing.relationship == edge.relationship
                for existing in self._edges
            )
            if not exists:
                self._edges.append(edge)
                self._adjacency[edge.source_id].append(edge)
        return hydrated

    def neighbors(self, node_id: str, *, scope: MemoryScope, depth: int = 1) -> list[MemoryNode]:
        if depth <= 0:
            return []
        edges = self.list_edges(scope=scope)
        adjacency: dict[str, list[MemoryEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.source_id].append(edge)

        seen = {node_id}
        frontier = [node_id]
        results: list[MemoryNode] = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for source in frontier:
                for edge in adjacency.get(source, []):
                    if edge.target_id in seen:
                        continue
                    seen.add(edge.target_id)
                    next_frontier.append(edge.target_id)
                    node = self.get_node(edge.target_id, scope=scope)
                    if node:
                        results.append(node)
            frontier = next_frontier
            if not frontier:
                break
        return results

    async def ensure_embedding(
        self,
        node: MemoryNode,
        *,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        if node.embedding is not None or embedder is None:
            return
        node.embedding = await embedder.embed_text(node.compressed)
        self._vector_store.upsert(node.id, node.embedding, scope=node.scope)
        self._repository.upsert_node(scope=node.scope, node=node)

    async def add_node_with_embedding(
        self,
        node: MemoryNode,
        *,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        await self.ensure_embedding(node, embedder=embedder)
        self.add_node(node)

    def set_node_index_state(
        self,
        *,
        scope: MemoryScope,
        node_id: str,
        status: Literal["pending", "indexed", "failed"],
        index_version: int | None = None,
    ) -> None:
        node = self._repository.get_node(scope=scope, node_id=node_id) or self.get_node(
            node_id,
            scope=scope,
        )
        if node is not None:
            node.index_status = status
            if index_version is not None:
                node.index_version = index_version
            self._nodes[node.id] = node
            self._repository.upsert_node(scope=scope, node=node)
        self._repository.set_node_index_state(
            scope=scope,
            node_id=node_id,
            status=status,
            index_version=index_version,
        )

    def search_lexical(
        self,
        query: str,
        *,
        scope: MemoryScope,
        top_k: int,
        memory_type: str | None = None,
        resolution: str = "compressed",
    ) -> list[tuple[MemoryNode, float]]:
        terms = {token for token in query.lower().split() if token}
        scored: list[tuple[MemoryNode, float]] = []
        for node in self.list_nodes(scope=scope):
            if memory_type and node.memory_type != memory_type:
                continue
            text = node_text_for_retrieval(node, resolution=resolution).lower()
            if not text:
                continue
            overlap = sum(1 for token in terms if token in text)
            recency_boost = 0.05 if node.last_accessed_at else 0.0
            score = float(overlap) + recency_boost + float(node.importance_score) * 0.01
            if score > 0:
                scored.append((node, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        for node, _ in scored[:top_k]:
            node.last_accessed_at = datetime.now(timezone.utc)
            self._repository.upsert_node(scope=node.scope, node=node)
        return scored[:top_k]

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        top_k: int,
        memory_type: str | None = None,
        threshold: float = 0.0,
        embedder: EmbedderProtocol | None = None,
        include_neighbors: bool = False,
        neighbor_depth: int = 1,
        resolution: str = "compressed",
    ) -> list[tuple[MemoryNode, float]]:
        """Search nodes by vector similarity when available, else lexical overlap."""
        if embedder is None:
            results = self.search_lexical(
                query,
                scope=scope,
                top_k=top_k,
                memory_type=memory_type,
                resolution=resolution,
            )
            return [(node, score) for node, score in results if score >= threshold]

        query_embedding = await embedder.embed_text(query)
        vector_hits = self._vector_store.search(query_embedding, scope=scope, top_k=max(top_k, 1) * 3)
        scored: list[tuple[MemoryNode, float]] = []
        for node_id, score in vector_hits:
            node = self.get_node(node_id, scope=scope)
            if node is None:
                continue
            if memory_type and node.memory_type != memory_type:
                continue
            if score >= threshold:
                node.last_accessed_at = datetime.now(timezone.utc)
                self._repository.upsert_node(scope=node.scope, node=node)
                scored.append((node, score))

        if not scored:
            scored = self.search_lexical(
                query,
                scope=scope,
                top_k=top_k,
                memory_type=memory_type,
                resolution=resolution,
            )

        if include_neighbors:
            scored = self._expand_with_neighbors(scored, scope=scope, depth=neighbor_depth)

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _expand_with_neighbors(
        self,
        scored: Iterable[tuple[MemoryNode, float]],
        *,
        scope: MemoryScope,
        depth: int,
    ) -> list[tuple[MemoryNode, float]]:
        seen: set[str] = set()
        expanded: list[tuple[MemoryNode, float]] = []
        for node, score in scored:
            if node.id in seen:
                continue
            seen.add(node.id)
            expanded.append((node, score))
            for neighbor in self.neighbors(node.id, scope=scope, depth=depth):
                if neighbor.id in seen:
                    continue
                seen.add(neighbor.id)
                expanded.append((neighbor, max(0.01, score * 0.8)))
        return expanded

