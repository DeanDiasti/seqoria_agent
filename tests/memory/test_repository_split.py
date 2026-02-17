import pytest

from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig
from seqoria_agent.memory.embeddings.base import EmbedderProtocol
from seqoria_agent.memory.graph.store import MemoryGraph
from seqoria_agent.memory.models import MemoryNode, MemoryScope
from seqoria_agent.memory.stores.in_memory import InMemoryMemoryRepository, InMemoryVectorStore


class _StaticEmbedder(EmbedderProtocol):
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed_text(self, text: str) -> list[float]:
        return list(self._vector)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class _FlakyEmbedder(EmbedderProtocol):
    def __init__(self) -> None:
        self.calls = 0

    async def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient failure")
        return [0.9, 0.1, 0.2]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(text) for text in texts]


@pytest.mark.asyncio
async def test_memory_graph_hydrates_vector_hits_from_repository():
    repo = InMemoryMemoryRepository()
    vectors = InMemoryVectorStore()
    scope = MemoryScope(tenant_id="tenant-a", user_id="user-1", conversation_id="conv-1")
    node = MemoryNode(
        id="node-a",
        memory_type="fact",
        compressed="remember apples",
        scope=scope,
        embedding=[0.1, 0.9],
    )
    repo.upsert_node(scope=scope, node=node)
    vectors.upsert(node.id, node.embedding or [0.0, 0.0], scope=scope)

    graph = MemoryGraph(repository=repo, vector_store=vectors)
    results = await graph.search(
        "apples",
        scope=scope,
        top_k=3,
        embedder=_StaticEmbedder([0.1, 0.9]),
    )

    assert [item.id for item, _ in results] == ["node-a"]


@pytest.mark.asyncio
async def test_dual_write_indexing_progresses_from_pending_to_indexed():
    embedder = _FlakyEmbedder()
    memory = AgenticMemory(embedder=embedder)

    await memory.start()
    node = await memory.add_memory("remember this fact", memory_type="fact")
    pending_node = memory.get_memory(node.id)
    assert pending_node is not None
    assert pending_node.index_status in {"pending", "indexed"}
    await memory.stop()

    indexed_node = memory.get_memory(node.id)
    assert indexed_node is not None
    assert indexed_node.index_status == "indexed"
    assert indexed_node.embedding is not None
    assert embedder.calls >= 2


@pytest.mark.asyncio
async def test_canonical_repository_and_vector_layer_are_scope_isolated():
    shared_repo = InMemoryMemoryRepository()
    shared_vectors = InMemoryVectorStore()
    cfg_a = AgenticMemoryConfig(tenant_id="tenant-a", user_id="user-1", conversation_id="conv-1")
    cfg_b = AgenticMemoryConfig(tenant_id="tenant-b", user_id="user-1", conversation_id="conv-1")

    mem_a = AgenticMemory(config=cfg_a, memory_repository=shared_repo, vector_store=shared_vectors)
    mem_b = AgenticMemory(config=cfg_b, memory_repository=shared_repo, vector_store=shared_vectors)

    node_a = await mem_a.add_memory("tenant a likes apples", memory_type="fact")
    node_b = await mem_b.add_memory("tenant b likes oranges", memory_type="fact")

    results_a = await mem_a.search_memories("apples", top_k=5)
    results_b = await mem_b.search_memories("oranges", top_k=5)

    assert [node.id for node in results_a] == [node_a.id]
    assert [node.id for node in results_b] == [node_b.id]
