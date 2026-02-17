from __future__ import annotations

from collections import defaultdict

from seqoria_agent.memory.models import MemoryEdge


def build_adjacency(edges: list[MemoryEdge]) -> dict[str, list[MemoryEdge]]:
    adjacency: dict[str, list[MemoryEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_id].append(edge)
    return adjacency

