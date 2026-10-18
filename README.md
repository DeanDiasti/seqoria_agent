# Seqoria Agent

An async Python library for building assistants that call Python tools and remote MCP servers. Use it inside a web application, run model/tool loops, stream responses, and keep conversation state under your application's control.

[PyPI](https://pypi.org/project/seqoria-agent/) · **Alpha API** · **Python 3.12+** · **Apache-2.0**

## Supported features

- Automatic model/tool loops, streaming responses, and single-step helpers for application-controlled loops.
- In-process tools with Pydantic argument validation, nested schemas, and hidden parameters supplied from trusted request context.
- MCP connections over Streamable HTTP, SSE, and stdio, plus a `SeqoriaMCP` server wrapper.
- Parallel tool execution with shared concurrency limits, timeouts, dependency scheduling, and structured failure events.
- OpenAI, Anthropic, Ollama, and OpenAI-compatible HTTP providers; optional MLX inference on Apple Silicon.
- Explicit conversation histories, provider fallback, and opt-in tool-result caching.

## Installation

```bash
pip install seqoria-agent
```

The package is named `seqoria-agent`; import it as `seqoria_agent`. HTTP providers use HTTPX and are included in the core installation. Vendor SDKs, model runtimes, and databases are not required for HTTP-provider usage.

Optional extras:

| Extra | Purpose |
| --- | --- |
| `redis` | Redis tool-result cache |
| `performance` | Optional JSON serialization and hashing accelerators |
| `server` | FastAPI dependency for application integration |
| `mlx-lm` | Local text inference on Apple Silicon |
| `mlx-vlm` | Local vision inference on Apple Silicon |
| `all` | All portable extras; MLX is installed separately |
| `dev,docs` | Contributor and documentation tools |

For example: `pip install 'seqoria-agent[redis,server]'`.

## Quickstart

Set `OPENAI_API_KEY` and set `OPENAI_MODEL` to a model ID available to your account. The example makes hosted model calls.

```python
import asyncio
import os

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models import AssistantMessage
from seqoria_agent.providers import AsyncOpenAIProvider

tools = LocalToolbox()


@tools.tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main():
    history = []
    async with AsyncOpenAIProvider(model_id=os.environ["OPENAI_MODEL"]) as provider:
        async with AsyncAIClient(
            providers=[provider],
            local_toolbox=tools,
            max_concurrency=16,
            tool_timeout=30,
        ) as client:
            async for event in client.chat("What is 2 + 3?", history=history):
                if isinstance(event, AssistantMessage):
                    print(event.content)

            # Reuse this history for another turn in the same conversation.
            async for event in client.chat("Now add 10 to that.", history=history):
                if isinstance(event, AssistantMessage):
                    print(event.content)


if __name__ == "__main__":
    asyncio.run(main())
```

The client records user messages, paired tool calls/results, and final assistant messages in the supplied history. Providers belong to the application; close clients before closing the providers they use.

For an example requiring no API key or service, run `python examples/manual_loop/basic_once.py` from a checkout or source distribution after installing the package.

## Providers

Import provider classes from `seqoria_agent.providers`:

| Backend | Async provider | Installation |
| --- | --- | --- |
| OpenAI | `AsyncOpenAIProvider` | Core |
| OpenAI-compatible endpoint | `AsyncOpenAICompatibleProvider` | Core |
| Anthropic | `AsyncAnthropicProvider` | Core |
| Ollama | `AsyncOllamaProvider` | Core; requires a running Ollama service |
| MLX text | `AsyncMLXLMProvider` | `mlx-lm` extra; Apple Silicon |
| MLX vision | `AsyncMLXVLMProvider` | `mlx-vlm` extra; Apple Silicon |

Use `Async*Provider` classes with `AsyncAIClient`. Blocking scripts can use `AIClient` as a regular context manager with either sync or async providers; `chat_sync()` returns the final answer. Provider classes without the `Async` prefix are the synchronous variants.

With multiple providers configured, the client can fall back after a provider error before streaming output begins. Errors after partial output propagate, avoiding replay of text or tool calls. Endpoint capabilities and available models depend on the backend.

## Streaming and tool failures

Inside an active client lifespan, use `stream_chat()` to receive text deltas, grouped tool events, and a final assistant message:

```python
from seqoria_agent.models import AssistantMessage, ChatChunk, ToolCallError, ToolMessages

async for event in client.stream_chat("What is 4 + 5?", history=history):
    if isinstance(event, ChatChunk):
        print(event.content, end="", flush=True)
    elif isinstance(event, ToolMessages):
        for message in event:
            if isinstance(message, ToolCallError):
                print(f"\nTool {message.name} failed: {message.error}")
    elif isinstance(event, AssistantMessage):
        # The final message contains the complete response and aggregate usage.
        # Avoid printing its content again after rendering the text deltas.
        if event.usage is not None:
            print("\nTokens:", event.usage.total_tokens)
```

Tool failures, including missing hidden parameters, become `ToolCallError` events. Independent calls can finish; declared dependents are skipped after a prerequisite fails. Provider and transport startup failures can still raise exceptions.

For application-controlled loops, use `llm_generate_once()` or `llm_stream_once()`, then execute requested tools through `run_tool_calls()` or `run_call_stream()`. `run_tool_calls()` preserves input order; `run_call_stream()` yields `(request, result)` pairs in completion order. Single-step helpers leave response recording to the caller.

## Serving multiple users

Create one `AsyncAIClient` per application worker and fixed tool/authentication configuration during application startup. Reuse its connections across independent conversations. Create resources within each worker's running event loop.

Give every conversation its own history list. Pass identity derived from authentication through `context` on each request. Hidden local tool arguments are omitted from the model's tool schema:

```python
@tools.tool(server_populated=["tenant_id", "user_id"])
async def account_summary(tenant_id: str, user_id: str) -> dict:
    """Read the authenticated user's account."""
    return await account_service.read(tenant_id=tenant_id, user_id=user_id)


# Application integration: authenticate, authorize conversation ownership,
# and acquire the conversation lock before running this turn.
async for event in client.chat(
    request_text,
    history=conversation_history,
    context={"tenant_id": principal.tenant_id, "user_id": principal.user_id},
):
    ...
```

`account_service`, `principal`, history loading, locking, and response delivery are application integration points. The application owns authentication, authorization, persistence, context budgeting, retention, rate limits, and serialization of simultaneous turns in one conversation. Multiple workers need storage transactions or distributed coordination when updating the same conversation.

Save model messages as JSON-compatible dictionaries:

```python
saved_history = [
    message if isinstance(message, dict) else message.model_dump(mode="json")
    for message in conversation_history
]
```

Those dictionaries can be loaded into a new history list for later turns. Reconcile incomplete history after a cancelled or failed turn before retrying; a tool's side effects may already have occurred.

Configure providers, credentials, remote servers, and tool blocklists for the client's whole lifespan. Remote MCP connections authenticated as different end users need separate clients/sessions. Remote servers resolve hidden identity from their authenticated server context; local `server_args` are not transmitted to remote tools.

### Execution limits

The default tool concurrency limit is 16 per client. `max_concurrency` and `tool_timeout` cover parameter resolution, cache I/O, and tool invocation. The timeout starts after acquiring a slot, so time spent waiting in the queue is excluded. Use `max_turns` to bound model/tool rounds and application-level limits for total request time and queue depth.

Synchronous tools and parameter callbacks run in worker threads. Cancellation cannot stop an already-running Python thread; bound underlying I/O and use async tools for cancellable work. CPU-intensive tasks belong in a process pool or job queue. Close abandoned async streams explicitly, for example with `contextlib.aclosing`.

### Caching and retries

Caching is disabled by default. Enable it only for tool collections whose successful results are safe to reuse:

```python
from seqoria_agent.client.cache import DictCache

cache = DictCache(max_entries=1024)
# Supply cache=cache and cache_ttl=60 when constructing the client.
```

Keys include arguments, injected values, trusted request context, and a per-client namespace. Include every authorization scope that affects a result in trusted context. `DictCache` copies mutable values and uses bounded LRU eviction with TTLs. `RedisCache` provides an external backend; close it with `await cache.aclose()`. Independently constructed clients do not share entries under the current namespace policy.

Transport retries default to one attempt. Additional attempts can repeat side effects after an ambiguous failure; enable retries only where replay is safe. Dependency declarations schedule calls requested in the same batch; authorize operations inside the tools themselves.

## Remote MCP tools

Expose tools with `SeqoriaMCP`, which extends the official MCP Python SDK's `FastMCP`:

```python
from seqoria_agent import SeqoriaMCP

server = SeqoriaMCP("Example", host="127.0.0.1", port=8000)


@server.tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    server.run(transport="streamable-http")
```

Connect using `AsyncAIClient(providers=[provider], mcp_servers=["http://127.0.0.1:8000/mcp"])`. Use `MCPServerConfig` from `seqoria_agent.client` for headers, timeouts, transport selection, stdio configuration, or OAuth. The default interactive OAuth flow is for local scripts; web applications supply their own authorization flow. Remote servers remain responsible for authentication and authorization.

## Upgrading from 0.6

Version 0.7 removed automatic memory and planning, background capture, embeddings, graph/vector/database adapters, and their extras. Replace memory objects with application-owned history lists and remove `retrieval_query` arguments. The library does not migrate or delete existing memory data.

`SeqoriaMCP` is the current server entry point; older FastXMCP examples are retired. Version 0.7.1 repairs failed-tool continuation, execution limits, cancellation, resource lifetimes, and examples. The API remains alpha; hosted integrations and production throughput require validation with your application's workload.

## Development and documentation

From a checkout or extracted source distribution:

```bash
python -m pip install -e '.[dev,docs]'
ruff check .
python -m pytest
python -m build
python -m twine check --strict dist/*
mkdocs build --strict
```

CI runs Python 3.12/3.13 on Linux, macOS, and Windows, including package build and core-import checks. Tests use mocked providers, in-memory MCP sessions, and a local MCP subprocess. Hosted models, model downloads, and external Redis are not required for the default suite.

The source distribution includes these guides and examples:

- [Web application lifespans and conversation ownership](docs/topics/multi_tenancy.md)
- [Tool execution and manual loops](docs/guides/ai_client.md)
- [MCP server configuration](docs/servers.md)
- [Caching](docs/topics/caching.md)
- [Release review and integration boundaries](docs/release-review.md)
- [Changelog](CHANGELOG.md) and [publishing guide](RELEASING.md)

## License

Apache-2.0. See [LICENSE.txt](LICENSE.txt).
