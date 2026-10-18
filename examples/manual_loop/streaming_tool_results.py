from __future__ import annotations

"""Demo script that shows how to consume tool results as they arrive.

Run with:

```bash
python examples/manual_loop/streaming_tool_results.py
```

"""

import asyncio
import datetime as _dt
import os
import random
from typing import List

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.models.chunk import ToolCallRequest
from seqoria_agent import AsyncOpenAIProvider


# ---------------------------------------------------------------------------
# Inline tools – one slow, one fast
# ---------------------------------------------------------------------------

tb = LocalToolbox()


@tb.tool()
async def slow_timestamp():
    """Return the timestamp after a small delay to simulate slow work."""
    await asyncio.sleep(1)
    return _dt.datetime.now().isoformat()


@tb.tool()
async def random_int():
    """Return a random integer immediately (fast)."""
    return random.randint(0, 100)


async def main() -> None:  # noqa: D401 – simple demo
    async with (
        AsyncOpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")) as provider,
        AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=tb) as ai,
    ):
        tool_calls: List[ToolCallRequest] = [
            ToolCallRequest(id="1", name="slow_timestamp", arguments={}),
            ToolCallRequest(id="2", name="random_int", arguments={}),
        ]

        print("Streaming tool execution - results will appear as soon as each tool finishes:\n")
        async for req, result in ai.run_call_stream(tool_calls=tool_calls):
            print(f"• {req.name} (id={req.id}) → {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
