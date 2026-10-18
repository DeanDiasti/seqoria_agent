# Seqoria Agent

An async Python library for building assistants that call Python tools and remote MCP servers. Use it inside a web application, run model/tool loops, stream responses, and keep conversation state under your application's control.

[PyPI](https://pypi.org/project/seqoria-agent/) · **Alpha API** · **Python 3.12+** · **Apache-2.0**

## Supported features

- Automatic model/tool loops, streaming responses, and single-step helpers for application-controlled loops.
- In-process tools with Pydantic argument validation, nested schemas, and hidden parameters supplied from trusted request context.
- MCP client connections over Streamable HTTP, SSE, and stdio.
- Parallel tool execution with shared concurrency limits, timeouts, dependency scheduling, and structured failure events.
- OpenAI, Anthropic, Ollama, and OpenAI-compatible HTTP providers.
- Explicit conversation histories and provider fallback.

## Installation

```bash
pip install seqoria-agent
```

The package is named `seqoria-agent`; import it as `seqoria_agent`. HTTP providers use HTTPX and are included in the core installation. Vendor SDKs, model runtimes, and databases are not required for HTTP-provider usage.

Optional extras:

| Extra | Purpose |
| --- | --- |
| `performance` | Optional JSON serialization and hashing accelerators |
| `server` | FastAPI dependency for application integration |
| `all` | All runtime extras |
| `dev,docs` | Contributor and documentation tools |

For example: `pip install 'seqoria-agent[server]'`.

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

The default tool concurrency limit is 16 per client. `max_concurrency` and `tool_timeout` cover parameter resolution and tool invocation. The timeout starts after acquiring a slot, so time spent waiting in the queue is excluded. Use `max_turns` to bound model/tool rounds and application-level limits for total request time and queue depth.

Synchronous tools and parameter callbacks run in worker threads. Cancellation cannot stop an already-running Python thread; bound underlying I/O and use async tools for cancellable work. CPU-intensive tasks belong in a process pool or job queue. Close abandoned async streams explicitly, for example with `contextlib.aclosing`.

### Tool execution and retries

Every requested tool call executes; the library does not cache or replay tool results. If your application needs caching, implement it within the operation or data service that owns the result, with explicit authorization scope, invalidation, and side-effect rules.

Transport retries default to one attempt. Additional attempts can repeat side effects after an ambiguous failure; enable retries only where replay is safe. Dependency declarations schedule calls requested in the same batch; authorize operations inside the tools themselves.

## Remote MCP tools

Connect to an existing MCP server during the client lifespan:

```python
from seqoria_agent.client import MCPServerConfig

server = MCPServerConfig(
    url="http://127.0.0.1:8000/mcp",
    timeout=30,
)
# Add mcp_servers=[server] when constructing AsyncAIClient.
# Enter the client context before discovering or calling its remote tools.
```

Use `MCPServerConfig` for headers, timeouts, transport selection, stdio configuration, or OAuth. The default interactive OAuth flow is for local scripts; web applications supply their own authorization flow. Remote servers remain responsible for authentication, authorization, and identity injection.

Seqoria Agent provides MCP client orchestration and `LocalToolbox` for in-process tools. Build and deploy remote servers separately with the official MCP SDK or another compatible implementation. The example servers under `examples/server-client/` use the official SDK directly.

## Migration

### From 0.8 to 0.9

- MLX providers and the `mlx-lm`/`mlx-vlm` extras are removed. For local inference, use Ollama or an OpenAI-compatible HTTP service.
- Use `context` and `param_registry` with `AsyncAIClient` for hidden local parameters. `LocalToolbox.register_param_provider` is removed; direct toolbox calls supply trusted `server_args` explicitly.
- Tool output stays intact: JSON keys such as `ai_message` and `metadata` are ordinary data. Interpret display metadata in your application. XML parsing helpers and the executor's `on_metadata` callback are removed.
- Use `mcp_servers` in client constructors and `ClientSettings`; the `urls` alias and unused `ClientSettings.client_kwargs` field are removed. Configure transports with `MCPServerConfig`.
- Use `ToolCallError.error` for failures and `ToolCallResult.result` for successes. The error's `result` alias is removed.
- Use transport/pool context managers for resource ownership. The no-op `MCPConnection.close()`, global HTTP pool helpers, and `ConnectionPool` alias are removed; use `MCPConnectionPool`.

### From 0.7 to 0.8

Version 0.8.0 removes `SeqoriaMCP`, `seqoria_agent.server`, and the server-only `XMCPError` exception. Migrate server code to an independent MCP server implementation. The client-side `mcp_servers` option and `LocalToolbox` remain supported.

Built-in tool-result caching is also removed. Drop `cache` and `cache_ttl` client/settings options, cache backend imports, and the `redis` extra. Implement any required caching within application-owned tools or data services.

### From 0.6

Version 0.7 removed automatic memory and planning, background capture, embeddings, graph/vector/database adapters, and their extras. Replace memory objects with application-owned history lists and remove `retrieval_query` arguments. The library does not migrate or delete existing memory data.

The API remains alpha; hosted integrations and production throughput require validation with your application's workload.

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

CI runs Python 3.12/3.13 on Linux, macOS, and Windows, including package build and core-import checks. Tests use mocked providers, in-memory MCP sessions, and a local MCP subprocess. Hosted models and model downloads are not required for the default suite.

The source distribution includes these guides and examples:

- [Web application lifespans and conversation ownership](docs/topics/multi_tenancy.md)
- [Tool execution and manual loops](docs/guides/ai_client.md)
- [MCP client connections](docs/servers.md)
- [Release review and integration boundaries](docs/release-review.md)
- [Changelog](CHANGELOG.md) and [publishing guide](RELEASING.md)

## License

Apache-2.0. See [LICENSE.txt](LICENSE.txt).
