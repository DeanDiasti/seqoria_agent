"""Unit tests for low-level value objects in *xmcp.models*."""

from seqoria_agent.models.chunk import ChatChunk


def test_chat_chunk_dataclass():
    """Ensure ChatChunk model round-trips basic fields after refactor."""

    chunk = ChatChunk(role="assistant", content="hi", model="stub")

    assert chunk.role == "assistant"
    assert chunk.content == "hi"
    assert chunk.model == "stub"
    # usage should be optional and None by default
    assert chunk.usage is None 