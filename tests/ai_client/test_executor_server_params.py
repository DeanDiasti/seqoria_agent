import pytest

from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.schemas import ToolCall


@pytest.mark.asyncio
async def test_missing_server_params_raises():  # noqa: D401
    box = LocalToolbox()

    @box.tool(server_populated=["tenant"])  # type: ignore[arg-type]
    def ping(tenant: str):  # noqa: D401, ANN001
        return f"pong-{tenant}"

    shared = {}
    pool = ConnectionPool(urls=[], tool_map=shared)
    registry = ToolRegistry(pool=pool, local_toolboxes=[box], tool_map=shared, refresh_interval=0)
    await registry.refresh()

    executor = Executor(pool=pool, registry=registry, allow_override=False)

    tc = ToolCall(id="p1", function={"name": "ping", "arguments": "{}"})

    with pytest.raises(RuntimeError):
        await executor._run_tool(tc, history=[]) 