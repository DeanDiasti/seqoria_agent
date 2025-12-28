# Streaming Responses

`AIClient.stream_chat()` yields token-by-token output so you can render assistant responses progressively.

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
            # handle other chunk types here
        print()  # newline after streaming

asyncio.run(main())
```

Under the hood:

1. The first LLM provider that successfully starts streaming is chosen.
2. Text deltas (`str`) are yielded immediately.
3. When the provider closes the stream, `AIClient` checks for tool calls and may perform additional loops.
4. After tool execution completes, the generator yields the final `LLMResponse` object.

---

## Chunk Schema

```python
ChatChunk(
    type="llm_delta",
    text: str | None,
    envelope: dict | None,
)
```

Only the `text` field is populated for `llm_delta` chunks. 