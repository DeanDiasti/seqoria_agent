import pytest

from seqoria_agent.client import AIClient  # type: ignore
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import ToolCall


class DummyProvider(BaseProvider):
    def __init__(self):
        pass

    async def async_generate(self, *args, **kwargs):  # pragma: no cover – not used
        raise NotImplementedError

    async def async_stream(self, *args, **kwargs):  # pragma: no cover – not used
        raise NotImplementedError


@pytest.mark.asyncio
async def test_run_tool_calls_mock(monkeypatch):  # noqa: D401
    """Wrapper must propagate whatever the underlying executor returns."""

    ai = AIClient(urls=[], providers=[DummyProvider()])

    async with ai as client:
        payload = {"answer": "cached", "display": None}

        async def fake_run_calls(*args, **kwargs):  # noqa: D401
            return payload

        # Monkeypatch the executor method
        monkeypatch.setattr(client.executor, "run_calls", fake_run_calls, raising=True)

        tc = ToolCall(id="abc", function={"name": "noop", "arguments": {}})
        out = await client.run_tool_calls(tool_calls=[tc], history=[])
        assert out is payload 