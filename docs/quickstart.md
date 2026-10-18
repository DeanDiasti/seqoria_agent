# Quickstart

## Try the offline example

From a checkout or extracted source distribution, install the package and run:

```bash
python -m pip install -e .
python examples/quickstart/offline.py
```

This deterministic example needs no API key or service. See the [example catalog](https://github.com/DeanDiasti/seqoria_agent/blob/main/examples/README.md) for prerequisites and focused examples.

## Connect a hosted model

Install with `pip install seqoria-agent` and set `OPENAI_API_KEY` and `OPENAI_MODEL` in your environment. HTTP providers are included in the core package. The following code makes hosted model calls:

```python
import asyncio
import os
from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.providers import AsyncOpenAIProvider
from seqoria_agent.models import AssistantMessage

tools = LocalToolbox()


@tools.tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main():
    history = []
    async with AsyncOpenAIProvider(model_id=os.environ["OPENAI_MODEL"]) as provider:
        async with AsyncAIClient(providers=[provider], local_toolbox=tools) as client:
            async for event in client.chat("What is 2 + 3?", history=history):
                if isinstance(event, AssistantMessage):
                    print(event.content)


asyncio.run(main())
```

`chat()` yields tool messages and the final assistant message. `stream_chat()` additionally yields `ChatChunk` text deltas; render deltas and use the final message for accounting rather than rendering its text a second time.

Use `Async*Provider` classes with `AsyncAIClient`. For synchronous scripts, use `AIClient` as a regular context manager and `chat_sync()` to return the final answer.

Reuse the history list for later turns in the same conversation. Keep histories separate across users and serialize simultaneous turns for one conversation. Close provider resources after closing the client.

Continue with [tool execution](guides/tool-execution.md) for event handling, [web applications](guides/web-applications.md) for shared clients and trusted identities, or [MCP connections](guides/mcp.md) for independently hosted tools.
