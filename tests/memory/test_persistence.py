from pathlib import Path

import pytest

from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig
from seqoria_agent.memory.stores.persistent import FilePersistenceStore


@pytest.mark.asyncio
async def test_persistence_round_trip_explicit_path(tmp_path: Path):
    memory = AgenticMemory()
    memory.begin_turn("Remember my favorite language is Python.")
    memory.commit_assistant_response("Got it, your favorite language is Python.")
    memory.end_turn()
    await memory.add_memory("Favorite language: Python", memory_type="fact")

    save_path = tmp_path / "memory.json"
    await memory.save(str(save_path))

    restored = await AgenticMemory.load(str(save_path))
    payload = restored.to_dict()
    assert payload["context"]["raw_messages"]
    assert payload["graph"]["nodes"]


@pytest.mark.asyncio
async def test_scope_based_path_generation(tmp_path: Path):
    store = FilePersistenceStore(base_dir=str(tmp_path))
    cfg_a = AgenticMemoryConfig(
        persist_memories=True,
        tenant_id="tenant-a",
        user_id="user-1",
        conversation_id="conv-1",
    )
    cfg_b = AgenticMemoryConfig(
        persist_memories=True,
        tenant_id="tenant-b",
        user_id="user-1",
        conversation_id="conv-1",
    )

    mem_a = AgenticMemory(config=cfg_a, persistence_store=store)
    mem_b = AgenticMemory(config=cfg_b, persistence_store=store)

    path_a = await mem_a.save(path=None)
    path_b = await mem_b.save(path=None)

    assert path_a != path_b
    assert Path(path_a).exists()
    assert Path(path_b).exists()

