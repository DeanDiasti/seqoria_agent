# Seqoria Agent

An async Python library for tool-using assistants in web applications. Connect remote MCP servers and in-process Python tools to OpenAI, Anthropic, Ollama, or an OpenAI-compatible endpoint.

**Status: alpha.** Python 3.12 or newer. Version 0.7 removes the automatic memory and planning subsystem. Conversation history belongs to your application.

## Install

```bash
pip install seqoria-agent
```

The core package includes HTTP-based providers. No OpenAI or Anthropic SDK, local model runtime, or database is required. Optional extras:

```bash
pip install 'seqoria-agent[redis]'        # Shared cache backend
pip install 'seqoria-agent[performance]'  # Faster JSON serialization and hashing
pip install 'seqoria-agent[server]'       # FastAPI integration
pip install 'seqoria-agent[mlx-lm]'       # Local text models, Apple Silicon only
pip install 'seqoria-agent[mlx-vlm]'      # Local vision models, Apple Silicon only
pip install 'seqoria-agent[dev,docs]'     # Contributor tools
```

## Quickstart

Set `OPENAI_API_KEY` and choose a model available to your account:

```python
import asyncio
import os

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models import ChatChunk
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
            async for event in client.stream_chat("What is 2 + 3?", history=history):
                if isinstance(event, ChatChunk):
                    print(event.content, end="", flush=True)

asyncio.run(main())
```

Async clients require `Async*Provider` classes. For a blocking script, use `AIClient` with a sync or async provider. Both clients append user, tool, and final assistant messages to the supplied history list. The single-step helpers let your application drive the loop explicitly.

## Serving multiple users

Create one `AsyncAIClient` per application worker and fixed tool/authentication configuration during application startup. Reuse it across requests. Give each conversation its own history list and pass authenticated identity through `context` on each request:

```python
@tools.tool(server_populated=["tenant_id", "user_id"])
async def account_summary(tenant_id: str, user_id: str) -> dict:
    """Read the authenticated user's account."""
    return await account_service.read(tenant_id=tenant_id, user_id=user_id)

# Inside a request handler, after authentication and authorization:
async for event in client.chat(
    request_text,
    history=conversation_history,
    context={"tenant_id": authenticated_tenant, "user_id": authenticated_user},
):
    ...
```

The application must authorize conversation access, serialize turns within a conversation, persist history, and enforce retention and request limits. Do not share user history or mutate a client's providers, credentials, or tool blocklist for individual requests. For MCP connections authenticated as different end users, use separate clients/sessions for those identities.

Synchronous Python tools run in worker threads. Use async tools for cancellable I/O; cancelling a request cannot terminate work already running in a Python thread. Tool concurrency limits apply per client in each worker. CPU-intensive work belongs in a process pool or job queue.

Caching is disabled by default. Enabling a cache caches successful tool results, so use it only for tool collections where replaying a cached result is correct. Entries include the full request context, injected arguments, and a per-client namespace. Supply all relevant authorization scope in trusted context. Remote servers must resolve identity from authenticated server context; client-side `server_args` are supported only for local tools.

See [the web application guide](https://github.com/DeanDiasti/seqoria_agent/blob/main/docs/topics/multi_tenancy.md) for a FastAPI lifespan example, and [the release guide](https://github.com/DeanDiasti/seqoria_agent/blob/main/RELEASING.md) for publishing.

## Remote MCP tools

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

Connect with `AsyncAIClient(providers=[provider], mcp_servers=["http://127.0.0.1:8000/mcp"])`. HTTP, SSE, and stdio transports are supported. For credentials, configure `MCPServerConfig` explicitly; remote servers remain responsible for authorization.

## Development

```bash
python -m pip install -e '.[dev,docs]'
ruff check .
python -m pytest
python -m build
python -m twine check --strict dist/*
mkdocs build --strict
```

Tests use mocked providers, an in-memory MCP session, and a local MCP subprocess. Live hosted models, local model runtimes, and external Redis services are not required.

## License

Apache-2.0; see [LICENSE.txt](https://github.com/DeanDiasti/seqoria_agent/blob/main/LICENSE.txt).
