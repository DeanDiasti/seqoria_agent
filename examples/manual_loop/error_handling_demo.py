from __future__ import annotations

"""Manual loop demo showcasing error handling via ToolCallError.

Run:

```bash
python examples/manual_loop/error_handling_demo.py
```
"""

import asyncio
import os
from typing import List


from seqoria_agent.client import AsyncAIClient
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.models.chunk import ToolCallRequest, ToolCallError, ToolCallResult
from seqoria_agent.providers import AsyncOpenAIProvider

# ---------------------------------------------------------------------------
# Inline tools – one succeeds, one fails
# ---------------------------------------------------------------------------

tb = LocalToolbox()


@tb.tool(description="Always raises an exception to demonstrate failure handling")
async def broken_tool():
    raise RuntimeError("Simulated tool failure - something went horribly wrong!")


@tb.tool(description="Returns a friendly string")
async def hello(name: str = "world") -> str:
    return f"Hello, {name}!"


async def main() -> None:  # noqa: D401 – simple demo
    async with (
        AsyncOpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")) as provider,
        AsyncAIClient(urls=[], providers=[provider], local_toolbox=tb) as ai,
    ):
        tool_calls: List[ToolCallRequest] = [
            ToolCallRequest(id="1", name="broken_tool", arguments={}),
            ToolCallRequest(id="2", name="hello", arguments={"name": "Seqoria"}),
        ]

        print("Streaming tool execution - note how failures are surfaced:\n")
        async for req, outcome in ai.run_call_stream(tool_calls=tool_calls):
            if isinstance(outcome, ToolCallError):
                print(f"✗ {req.name} (id={req.id}) FAILED → {outcome.error}")
            elif isinstance(outcome, ToolCallResult):
                print(f"✓ {req.name} (id={req.id}) → {outcome.result}")


if __name__ == "__main__":
    asyncio.run(main())
