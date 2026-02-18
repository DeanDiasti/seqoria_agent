import pytest

from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig


@pytest.mark.asyncio
async def test_context_assembly_orders_system_compressed_recent():
    config = AgenticMemoryConfig(total_context_budget=12000)
    memory = AgenticMemory(config=config)
    memory.add_system_message("Base system prompt")
    memory.add_user_message("Tell me about quarterly revenue.")
    memory.add_assistant_message("Q1 revenue was up by 8 percent.")
    memory._context_manager.set_compressed_history("Older summary here.")  # noqa: SLF001
    await memory.add_memory(
        "User frequently asks about revenue trends and quarterly deltas.",
        memory_type="fact",
    )

    messages = memory.to_messages(query="revenue trends")

    assert messages[0]["role"] == "system"
    assert "Base system prompt" in messages[0]["content"]
    assert "<relevant_memories>" in messages[0]["content"]
    assert messages[1]["role"] == "system"
    assert "<earlier_conversation>" in messages[1]["content"]
    assert any(m.get("role") == "user" for m in messages)
    assert any(m.get("role") == "assistant" for m in messages)


def test_update_system_prompt_replaces_base_prompt_in_context():
    memory = AgenticMemory(config=AgenticMemoryConfig(total_context_budget=12000))
    memory.add_system_message("Original system prompt")
    memory.update_system_prompt("Updated system prompt")

    messages = memory.to_messages()

    assert memory.system_prompt == "Updated system prompt"
    assert messages
    assert messages[0]["role"] == "system"
    assert "Updated system prompt" in messages[0]["content"]
    assert "Original system prompt" not in messages[0]["content"]


def test_update_system_prompt_can_clear_base_prompt():
    memory = AgenticMemory(config=AgenticMemoryConfig(total_context_budget=12000))
    memory.add_system_message("Prompt to clear")
    memory.add_user_message("hello")
    memory.update_system_prompt(None)

    messages = memory.to_messages()

    assert memory.system_prompt is None
    assert messages
    assert messages[0]["role"] == "user"

