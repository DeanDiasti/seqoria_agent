import pytest

from seqoria_agent.memory import AgenticMemory


@pytest.mark.asyncio
async def test_search_memories_returns_relevant_nodes():
    memory = AgenticMemory()
    apples = await memory.add_memory(
        "The user likes green apples and asks for apple recipes.",
        memory_type="fact",
    )
    await memory.add_memory(
        "The user is debugging postgres connection pooling issues.",
        memory_type="fact",
    )

    results = await memory.search_memories("apple recipes", top_k=2)
    assert results
    assert results[0].id == apples.id


@pytest.mark.asyncio
async def test_query_fallback_uses_latest_user_turn():
    memory = AgenticMemory()
    await memory.add_memory("The user prefers concise answers about cloud costs.", memory_type="fact")

    memory.begin_turn("Can you summarize cloud costs?")
    messages = memory.messages_for_llm()

    assert messages
    assert messages[0]["role"] == "system"
    assert "<relevant_memories>" in messages[0]["content"]

