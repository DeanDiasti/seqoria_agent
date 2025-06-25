# Using the AIClient

The `seqoria_agent.AIClient` is a powerful, asynchronous client that orchestrates interactions between an LLM and **remote XMCP servers, local inline tools, or both**.

## Initialization

You initialize the client with a list of server URLs and a list of LLM providers.

```python
import asyncio
from seqoria_agent import AIClient
from seqoria_agent.providers import OpenAIProvider, AnthropicProvider

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

`AIClient` surfaces several synchronous hooks so your application can react instantly:

* **`on_display`** – fires when a `DisplayEnvelope` arrives (UI rendering).
* **`on_error`** – fires on structured `<error>` payloads (alerting / retry).
* **`on_metadata`** – *new in v0.3* – fires when a tool response is wrapped in the JSON envelope.  All keys from `metadata` are passed unchanged so you can log execution time, tool name, etc.
* **`cache`** – any object implementing `get/set` avoids redundant tool calls.
* **`logger`** – plug in your own `structlog` or stdlib logger.

```python
from seqoria_agent.client.cache import RedisCache

def my_display_handler(envelope, history):
    print("Rendering UI for:", envelope['title'])

ai = AIClient(
    urls=[...],
    providers=[...],
    on_display=my_display_handler,
    cache=RedisCache(url="redis://localhost:6379"),
    on_metadata=lambda meta, _hist: print("[debug] metadata", meta),
)
```

## Local Inline Tools

Need zero-latency, in-process tools?  Wrap them in a `LocalToolbox` and pass it
to the client.  They are advertised to the LLM exactly like remote tools, but
executed locally:

```python
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.models.envelope import DisplayEnvelope

toolbox = LocalToolbox()

@toolbox.tool()
def add(a: int, b: int) -> int:
    return a + b

@toolbox.display_tool()
def shout(msg: str, show_user: bool = False):
    if show_user:
        return DisplayEnvelope(type="markdown", payload=f"## {msg.upper()} 🔊")
    return msg.upper()

async with AIClient(urls=[], providers=[OpenAIProvider(model_id="gpt-4o")],
                    local_toolbox=toolbox) as ai:
    res = await ai.chat("What is 2+3?", history=[])
    print(res["answer"])
```

## Pure LLM Mode (No Tools)

If you just need a chat wrapper around the provider, create the client with
**no URLs and no toolboxes**.  The client automatically skips the tool loop:

```python
async with AIClient(urls=[], providers=[OpenAIProvider(model_id="gpt-4o")]) as ai:
    res = await ai.chat("Explain quantum tunnelling in one paragraph", history=[])
    print(res["answer"])
```

## What's new in v0.2+

* **Dependency-graph execution** – automatic parallel layers respecting `depends_on`.
* **Structured error & display callbacks** – `on_error`, `on_display`.
* **Meta-tools** – `__think__`, `__summarize__`, `__clarify__`.
* **Retry / circuit-breaker** & **adaptive concurrency** (`max_concurrency`, `tool_timeout`).
* **Pluggable cache backend** – `DictCache`, `RedisCache`.
* **Local inline tools** – bring functions in-process with `LocalToolbox`.
* **Router merging** – compose servers with `FastXMCPRouter`.
* **LLM-only mode** – operate without any tools (`urls=[]`).
* **JSON wrapper support** – opt-in `json_wrapper=True` on servers; client auto-unwraps

## Quick Example (stateless)
```python
history = []
async with AIClient(urls=["http://localhost:3333"],
                    providers=[OpenAIProvider(model_id="gpt-4o")]) as ai:
    res = await ai.chat("Current time + divide 10 by 2", history)
    print(res["answer"], "display?", bool(res["display"]))
```

## Authentication

Need to call **protected** tools?  Pass an *AuthProvider* when you build the client.

```python
from seqoria_agent import AIClient
from seqoria_agent.auth.providers import StaticCookieProvider
from seqoria_agent.providers import OpenAIProvider

cookies = StaticCookieProvider(token=my_jwt, cookie_name="access_token")

async with AIClient(urls=["https://api.acme.ai/mcp/"],
                    providers=[OpenAIProvider(model_id="gpt-4o")],
                    auth_provider=cookies) as ai:
    res = await ai.chat("who am I?", history=[])
    print(res["answer"])
```

You may **replace** credentials without restarting the client:

```python
ai.add_auth_provider(StaticApiKeyProvider(token="new", header_name="X-API-Key"))
```

Auth providers can implement `refresh_token_if_needed()` for automatic renewal before each request.

> **Heads-up**  This page is a concise primer. A full deep-dive lives in [Guides → Using AIClient](guides/ai_client.md). 