from seqoria_agent.memory.graph.store import MemoryGraph
from seqoria_agent.memory.models import MemoryNode, MemoryScope


def test_graph_search_is_scope_isolated():
    graph = MemoryGraph()
    scope_a = MemoryScope(tenant_id="tenant-a", user_id="user-1", conversation_id="conv-1")
    scope_b = MemoryScope(tenant_id="tenant-b", user_id="user-1", conversation_id="conv-1")

    graph.add_node(
        MemoryNode(
            id="a1",
            memory_type="fact",
            compressed="tenant a likes apples",
            scope=scope_a,
        )
    )
    graph.add_node(
        MemoryNode(
            id="b1",
            memory_type="fact",
            compressed="tenant b likes apples",
            scope=scope_b,
        )
    )

    results_a = graph.search_lexical("apples", scope=scope_a, top_k=5)
    results_b = graph.search_lexical("apples", scope=scope_b, top_k=5)

    assert [node.id for node, _ in results_a] == ["a1"]
    assert [node.id for node, _ in results_b] == ["b1"]

