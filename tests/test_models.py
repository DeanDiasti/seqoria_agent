"""Unit tests for low-level value objects in *xmcp.models*."""

from seqoria_agent.models.chunk import ChatChunk
from seqoria_agent.models.envelope import DisplayEnvelope


def test_display_envelope_roundtrip():
    env = DisplayEnvelope(
        type="markdown",
        payload="**hello**",
        title="Greeting",
        meta={"lang": "en"},
    )
    dumped = env.dict_for_transport()
    # The dict should contain the same keys
    assert dumped["type"] == "markdown"
    assert dumped["payload"] == "**hello**"
    assert dumped["title"] == "Greeting"
    assert dumped["meta"] == {"lang": "en"}
    # Round-tripping through model_validate should succeed
    env2 = DisplayEnvelope.model_validate(dumped)
    assert env2 == env


def test_chat_chunk_dataclass():
    """Ensure ChatChunk model round-trips basic fields after refactor."""

    chunk = ChatChunk(role="assistant", content="hi", model="stub")

    assert chunk.role == "assistant"
    assert chunk.content == "hi"
    assert chunk.model == "stub"
    # usage should be optional and None by default
    assert chunk.usage is None 