import pytest

from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.schemas import ToolCall


@pytest.mark.asyncio
async def test_metadata_callback_triggered():  # noqa: D401
    # Local toolbox that wraps responses in JSON
    box = LocalToolbox(json_wrapper=True)

    @box.tool()
    def square(x: int) -> int:  # noqa: D401
        return int(x) ** 2

    shared = {}
    pool = ConnectionPool(urls=[], tool_map=shared)
    registry = ToolRegistry(pool=pool, local_toolboxes=[box], tool_map=shared, refresh_interval=0)
    await registry.refresh()

    captured_meta = {}

    def _on_meta(meta, hist):  # noqa: D401, ANN001
        captured_meta.update(meta)

    executor = Executor(pool=pool, registry=registry, on_metadata=_on_meta)

    tc = ToolCall(id="sq", function={"name": "square", "arguments": '{"x": 5}'})
    history = []

    _, hist_entry, _ = await executor._run_tool(tc, history)

    # Metadata should have been captured via callback
    assert captured_meta.get("tool_name") == "square"
    assert hist_entry["content"].startswith("<tool_response") 