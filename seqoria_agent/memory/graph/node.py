from __future__ import annotations

from seqoria_agent.memory.models import MemoryNode, MemoryScope


def node_matches_scope(node: MemoryNode, scope: MemoryScope) -> bool:
    return node.scope.matches(scope)


def node_text_for_retrieval(node: MemoryNode, resolution: str = "compressed") -> str:
    if resolution == "full" and node.full:
        return node.full
    if resolution == "hybrid" and node.hybrid:
        return node.hybrid
    return node.compressed

