import asyncio
import types

import pytest

from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.cache import DictCache
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.schemas import ToolCall


@pytest.mark.asyncio
async def test_executor_uses_cache():  # noqa: D401
    box = LocalToolbox()

    # Mutable container so we can swap implementation between calls
    state = {"value": 1}

    @box.tool()
    def inc() -> int:  # noqa: D401
        return state["value"]

    # Shared map
    tool_map = {}
    pool = ConnectionPool(urls=[], tool_map=tool_map)
    registry = ToolRegistry(pool=pool, local_toolboxes=[box], tool_map=tool_map, refresh_interval=0)
    await registry.refresh()

    cache = DictCache()
    executor = Executor(pool=pool, registry=registry, cache=cache, cache_ttl=5)

    tc = ToolCall(id="inc", function={"name": "inc", "arguments": "{}"})
    history: list[dict] = []

    # First execution – should call underlying function and store result
    _, first_hist, _ = await executor._run_tool(tc, history)
    assert "1" in first_hist["content"]

    # Mutate state so underlying function would now return different value
    state["value"] = 42

    # Second execution – cache should return old value (1), ignoring updated function
    _, second_hist, _ = await executor._run_tool(tc, history)
    assert "1" in second_hist["content"] 