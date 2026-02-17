import pytest

from seqoria_agent.memory import AgenticMemory


@pytest.mark.asyncio
async def test_resolutions_are_structurally_distinct():
    memory = AgenticMemory()
    node = await memory.add_memory(
        content=(
            "EMEA Q3 margin remained resilient despite cost pressure. "
            "Growth decelerated versus Q2, indicating possible market saturation. "
            "Recommendation: strengthen cost controls and test expansion options."
        ),
        memory_type="conversation",
        metadata={"turn": 1, "source": "test"},
    )

    assert node.compressed != node.hybrid
    assert node.hybrid != node.full
    assert "[conversation]" in node.compressed
    assert "<summary>" in (node.hybrid or "")
    assert "<structured_details>" in (node.full or "")


def test_context_window_builder_includes_multiple_messages():
    memory = AgenticMemory()
    memory.add_user_message("Pull EMEA Q3 data.")
    memory.add_assistant_message("I will call the sales tool and compute margin.")
    memory.add_tool_call(
        {"role": "tool_call", "id": "tc-1", "name": "sales_snapshot", "arguments": {"region": "EMEA"}}
    )

    window = memory._build_context_window_text(max_messages=3)  # noqa: SLF001
    assert "- user:" in window
    assert "- assistant:" in window
    assert "- tool_call:" in window

