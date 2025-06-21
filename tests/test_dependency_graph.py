import asyncio
import pytest

from seqoria_agent.client.client import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import ToolCall
from seqoria_agent.models.errors import DependencyError


class DummyProvider(BaseProvider):
    """Minimal provider stub that satisfies the abstract interface but is never used."""

    def __init__(self, model_id: str | None = None, **kwargs):  # noqa: D401
        self.model_id = model_id or "dummy"

    async def async_generate(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("DummyProvider should not be invoked in unit tests")

    async def async_stream(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("DummyProvider should not be invoked in unit tests")


@pytest.fixture()
def ai_client():
    # We do not enter the async context; _compute_execution_layers does not need it.
    return AIClient(urls=["http://dummy"], providers=[DummyProvider()])


def _tc(name: str) -> ToolCall:
    """Convenience helper to build a minimal ToolCall."""
    return ToolCall(id=f"call-{name}", function={"name": name, "arguments": "{}"})


def test_topological_order(ai_client):
    """A -> B -> C should yield three sequential layers."""

    ai_client._tool_deps = {  # type: ignore[attr-defined]
        "b": {"a"},
        "c": {"b"},
    }

    layers = ai_client._compute_execution_layers([_tc("a"), _tc("b"), _tc("c")])
    assert [[tc.function["name"] for tc in layer] for layer in layers] == [["a"], ["b"], ["c"]]


def test_parallel_layers(ai_client):
    """A and B independent, C depends on both → two layers: [A,B] then [C]."""

    ai_client._tool_deps = {
        "c": {"a", "b"},
    }

    layers = ai_client._compute_execution_layers([_tc("a"), _tc("b"), _tc("c")])
    names = [set(tc.function["name"] for tc in layer) for layer in layers]
    assert names[0] == {"a", "b"}
    assert names[1] == {"c"}


def test_cycle_detection(ai_client):
    """A<->B cycle should raise DependencyError."""

    ai_client._tool_deps = {
        "a": {"b"},
        "b": {"a"},
    }

    with pytest.raises(DependencyError):
        ai_client._compute_execution_layers([_tc("a"), _tc("b")]) 