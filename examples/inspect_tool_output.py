import asyncio
import json
import uuid

from seqoria_agent import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.server.formatters import make_formatter


# -----------------------------------------------------------------------------
# 1) Minimal dummy provider – we won't actually query an LLM, but AIClient
#    requires at least one provider instance. This stub fulfils the interface.
# -----------------------------------------------------------------------------


class DummyProvider(BaseProvider):
    """A no-op LLM provider that immediately returns an empty completion."""

    def __init__(self):
        # Nothing to initialise – no external credentials required.
        pass

    # Synchronous generation – return an empty response
    async def async_generate(self, *args, **kwargs):  # noqa: D401, ANN401
        return LLMResponse(model_id="dummy", text="(no llm response)")

    # Streaming – yield nothing but the final empty response so the caller can
    # iterate without hanging.
    async def async_stream(self, *args, **kwargs):  # noqa: D401, ANN401
        yield ""  # an empty delta to satisfy the iterator contract
        yield LLMResponse(model_id="dummy", text="(no llm response)")


# -----------------------------------------------------------------------------
# 2) Define a local toolbox with two example tools – one plain and one display
# -----------------------------------------------------------------------------

# Enable JSON wrapper so tool results are returned as
# {"ai_message": "<tool_response …/>", "metadata": {...}}
box = LocalToolbox(json_wrapper=True)


# -----------------------------------------------------------------------------
# 2a) The derive-formatter logic will automatically create a <sum> wrapper tag
# -----------------------------------------------------------------------------


@box.tool()
def add(a: int, b: int) -> int:
    """Return the arithmetic sum of *a* and *b*."""
    return a + b


# -----------------------------------------------------------------------------
# 2b) Multiply tool – demonstrates explicit formatter passed in ---------------
# -----------------------------------------------------------------------------

@box.tool(response_format=make_formatter(top_level_fields=["product"]))
def multiply(a: int, b: int):  # noqa: D401
    """Return a dict so the formatter emits <product> tag automatically."""
    return {"product": int(a) * int(b)}


# -----------------------------------------------------------------------------
# 2c) Greet tool – demonstrates a tool that returns a simple dict -------------
# -----------------------------------------------------------------------------

@box.tool()
def greet(name: str):
    """Return a greeting message."""
    return {"message": f"Hello {name}"}


# -----------------------------------------------------------------------------
# 3) Helper coroutine to demonstrate call_mcp_tool and _run_tool outputs
# -----------------------------------------------------------------------------


async def demo() -> None:
    async with AIClient(providers=[DummyProvider()], urls=[], local_toolbox=box) as client:
        print("\n--- call_mcp_tool (add) ------------------------------")
        wrapped = await client.call_mcp_tool(name="add", arguments={"a": 3, "b": 4})
        print("Raw transport payload:")
        print(wrapped)

        payload_dict = json.loads(wrapped[0].text)
        print("\nParsed wrapper → metadata:")
        print(payload_dict["metadata"])
        print("\nai_message XML:")
        print(payload_dict["ai_message"])

        print("\n--- _run_tool (add) ---------------------------------")
        tc_add = ToolCall(
            id=str(uuid.uuid4()),
            function={"name": "add", "arguments": json.dumps({"a": 1, "b": 2})},
        )
        rt_add = await client._run_tool(tc_add)
        print("History entry captured by AIClient:")
        print(rt_add[1])

        print("\n--- _run_tool (multiply) -----------------------------")
        tc_mul = ToolCall(
            id=str(uuid.uuid4()),
            function={"name": "multiply", "arguments": json.dumps({"a": 6, "b": 7})},
        )
        rt_mul = await client._run_tool(tc_mul)
        print(rt_mul)

        print("\n--- _run_tool (greet) ---------------------------------")
        tc_greet = ToolCall(
            id=str(uuid.uuid4()),
            function={
                "name": "greet",
                "arguments": json.dumps({"name": "Alice"}),
            },
        )
        rt_greet = await client._run_tool(tc_greet)
        print(rt_greet)


# -----------------------------------------------------------------------------
# 4) Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(demo()) 