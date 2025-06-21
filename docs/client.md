# Using the AIClient

The `xmcp.AIClient` is a powerful, asynchronous client for orchestrating interactions between an LLM and one or more `XMCP` servers.

## Initialization

You initialize the client with a list of server URLs and a list of LLM providers.

```python
import asyncio
from xmcp import AIClient
from xmcp.providers import OpenAIProvider, AnthropicProvider

# A list of providers to try in order
providers = [
    OpenAIProvider(model_id="gpt-4o"),
    AnthropicProvider(model_id="claude-3-haiku-20240307"),
]

async def main():
    async with AIClient(urls=["http://localhost:3333"], providers=providers) as ai:
        # ... use the client
        pass

asyncio.run(main())
```

The client will automatically connect to the servers and discover their available tools upon entering the context block.

## Chatting

The primary method is `chat()`, which takes a user query and a conversation history. The client is **stateless**, meaning you are responsible for maintaining the history object between turns.

```python
history = []
result = await ai.chat("What is the weather in London?", history=history)

# The 'result' dict contains the answer and the updated history
print(result["answer"])
history = result["history"] 
```

The client handles multi-turn conversations, automatically executing tools in parallel and feeding the results back to the LLM until it arrives at a final answer.

## Streaming

For a more responsive user experience, use `stream_chat()`. This method returns an async generator that yields `ChatChunk` objects as they become available.

```python
history = []
async for chunk in ai.stream_chat("Give me a sales summary for EMEA.", history=history):
    if chunk.type == "llm_delta":
        print(chunk.text, end="", flush=True)
    elif chunk.type == "tool_display":
        print("\n--- UI Component Received ---")
        render(chunk.envelope) # Your display logic
```

## Special Features

### Meta-Tools

The `AIClient` injects special client-side tools into the LLM's toolset to improve reasoning and flow control:

-   `__think__`: Allows the LLM to record its internal thoughts without showing them to the user.
-   `__summarize__`: Lets the LLM compress the conversation history to save tokens.
-   `__clarify__`: Enables the LLM to ask the user a follow-up question if a query is ambiguous.

These are handled automatically by the client.

### Hot-Reloading Tools

You can configure the client to periodically refresh its knowledge of available tools from the servers, enabling zero-downtime tool deployment.

```python
# Refresh tools every 5 minutes (300 seconds)
ai = AIClient(urls=[...], providers=[...], refresh_interval=300)

# Or trigger manually
await ai.refresh_tools()
```

### Callbacks and Hooks

The client provides hooks to integrate with your application's logic:

-   `on_display`: A callback that fires immediately when a `DisplayEnvelope` is received from a tool.
-   `on_error`: A callback for handling structured errors from tools.
-   `cache`: A pluggable cache object (e.g., `RedisCache`) to avoid redundant tool calls.
-   `logger`: A standard or `structlog` logger for observability.

```python
from xmcp.client.cache import RedisCache

def my_display_handler(envelope, history):
    print("Rendering UI for:", envelope['title'])

ai = AIClient(
    urls=[...],
    providers=[...],
    on_display=my_display_handler,
    cache=RedisCache(url="redis://localhost:6379")
)
```

## What's new in v0.2+

* **Dependency-graph execution** – automatic parallel layers respecting `depends_on`.
* **Structured error & display callbacks** – `on_error`, `on_display`.
* **Meta-tools** – `__think__`, `__summarize__`, `__clarify__`.
* **Retry / circuit-breaker** & **adaptive concurrency** (`max_concurrency`, `tool_timeout`).
* **Pluggable cache backend** – `DictCache`, `RedisCache`.

## Quick Example (stateless)
```python
history = []
async with AIClient(urls=["http://localhost:3333"],
                    providers=[OpenAIProvider(model_id="gpt-4o")]) as ai:
    res = await ai.chat("Current time + divide 10 by 2", history)
    print(res["answer"], "display?", bool(res["display"]))
```

> **Heads-up**  This page is a concise primer. A full deep-dive lives in [Guides → Using AIClient](guides/ai_client.md). 