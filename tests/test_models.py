"""Unit tests for low-level value objects in *xmcp.models*."""

from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.chunk import ChatChunk


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
    chunk1 = ChatChunk(type="llm_delta", text="hi")
    assert chunk1.type == "llm_delta"
    assert chunk1.text == "hi"
    assert chunk1.envelope is None

    env = {"type": "notification", "payload": "done"}
    chunk2 = ChatChunk(type="tool_display", envelope=env)
    assert chunk2.type == "tool_display"
    assert chunk2.envelope == env 