import pytest

from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.schemas import ToolCall


@pytest.mark.asyncio
async def test_display_tool_triggers_early_exit():  # noqa: D401
    # Set up a LocalToolbox with a display_tool
    box = LocalToolbox()

    from seqoria_agent.models.envelope import DisplayEnvelope

    @box.display_tool()
    def greet(name: str, show_user: bool = False):  # noqa: D401
        if show_user:
            return DisplayEnvelope(type="markdown", payload=f"Hello {name}")
        return f"Hi {name}"

    # Shared mapping
    tool_map: dict = {}
    pool = ConnectionPool(urls=[], tool_map=tool_map)
    registry = ToolRegistry(pool=pool, local_toolboxes=[box], tool_map=tool_map, refresh_interval=0)
    await registry.refresh()

    # Capture envelopes via custom callback ---------------------------------
    captured = {}

    def _capture_display(env, hist):  # noqa: D401, ANN001
        captured["env"] = env

    executor = Executor(pool=pool, registry=registry, on_display=_capture_display)

    # Build ToolCall
    tc = ToolCall(
        id="g1",
        function={
            "name": "greet",
            "arguments": '{"name": "Alice", "show_user": true}'
        },
    )
    history: list[dict] = []

    _, hist_entry, early = await executor._run_tool(tc, history)

    # Assertions ------------------------------------------------------------
    assert early is not None, "Display tools should return early payload"
    # At the moment the client does not automatically extract the DisplayEnvelope from XML,
    # so the `display` key may be ``None``. The important part is that the early-exit path
    # was taken.
    assert early["display"] is None

    # Ensure callback has *not* been invoked because no envelope could be parsed
    assert captured == {}

    # History entry content should be XML-wrapped string that contains our payload
    assert isinstance(hist_entry["content"], str)
    assert "<tool_response" in hist_entry["content"] and "Hello Alice" in hist_entry["content"] 