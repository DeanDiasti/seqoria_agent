# Streaming Responses

`AIClient.stream_chat()` gives you token-by-token output **and** forward DisplayEnvelope payloads as soon as they arrive.

```python
import asyncio
from xmcp import AIClient
from xmcp.providers import OpenAIProvider

async def main():
    async with AIClient(["http://localhost:3333"], [OpenAIProvider(model_id="gpt-4o")]) as ai:
        history = []
        async for chunk in ai.stream_chat("Render sales as a chart", history):
            if chunk.type == "llm_delta":
                print(chunk.text, end="", flush=True)
            elif chunk.type == "tool_display":
                ui.render(chunk.envelope)
        print()  # newline after streaming

asyncio.run(main())
```

Under the hood:

1. The first LLM provider that successfully starts streaming is chosen.
2. Text deltas (`str`) are yielded immediately.
3. When the provider closes the stream, `AIClient` checks for tool calls and may perform additional loops.
4. If a **display tool** returns an envelope, the generator yields `ChatChunk(type="tool_display", …)` and terminates early.

---

## Chunk Schema

```python
ChatChunk(
    type="llm_delta" | "tool_display",
    text: str | None,
    envelope: dict | None,
)
```

Only one of `text` or `envelope` is populated per chunk. 