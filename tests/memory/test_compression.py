from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig, ContextConfig
from seqoria_agent.memory.context.compressor import Compressor


def test_progressive_compression_and_checkpoint_trigger():
    config = AgenticMemoryConfig(
        context=ContextConfig(
            max_recent_messages=1,
            compression_enabled=True,
            compression_strategy="progressive",
            compress_after_turns=1,
            compress_after_tokens=1,
            checkpoint_every_n_turns=1,
        )
    )
    memory = AgenticMemory(config=config)

    memory.begin_turn("What happened in Q2?")
    memory.commit_assistant_response("Q2 improved margin by 2 points.")
    memory.end_turn()

    assert memory._context_manager.compressed_history is not None  # noqa: SLF001
    checkpoints = memory.list_checkpoints()
    assert checkpoints
    assert checkpoints[-1].turn_index >= 1


def test_default_compression_strategy_is_hierarchical():
    config = ContextConfig()
    assert config.compression_strategy == "hierarchical"


def test_hierarchical_compression_contains_two_stage_sections():
    compressor = Compressor(strategy="hierarchical")
    summary = compressor.compress(
        [
            {"role": "user", "content": "Find revenue trends for Q2 in EMEA and summarize risks."},
            {"role": "assistant", "content": "I found growth but also volatility in two regions."},
            {"role": "tool_result", "content": "Rows returned: 412"},
        ],
        existing_summary="Previous interactions focused on quarterly forecasting.",
    )

    assert "[INTERACTION SUMMARIES]" in summary
    assert "[USAGE SUMMARY]" in summary
    assert "[PREVIOUS USAGE SUMMARY]" in summary

