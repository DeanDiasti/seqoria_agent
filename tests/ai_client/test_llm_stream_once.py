import pytest

from seqoria_agent.client import AIClient  # type: ignore
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class StreamProvider(BaseProvider):
    """Provider that streams fixed characters then final response."""

    def __init__(self):
        pass

    async def async_generate(self, *args, **kwargs):  # pragma: no cover – not used
        raise NotImplementedError

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        async def _gen():
            for ch in "done":
                yield ch
            yield LLMResponse(model_id="stub", text="done", tool_calls=[])
        return _gen()


@pytest.mark.asyncio
async def test_llm_stream_once_llm_only():  # noqa: D401
    ai = AIClient(urls=[], providers=[StreamProvider()])
    async with ai as client:
        hist = client.prepare_history("ping", [])
        text_chunks = []
        final_text = None
        async for chunk in client.llm_stream_once(hist, use_tools=False):
            if isinstance(chunk, str):
                text_chunks.append(chunk)
            else:
                final_text = chunk.text

        assert "".join(text_chunks) == "done"
        assert final_text == "done" 