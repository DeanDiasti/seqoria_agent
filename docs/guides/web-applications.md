# Serving multiple users

Use one async client per application worker and fixed tool/authentication configuration. Give each conversation its own history and pass trusted identity separately on every tool-executing request. The library has no user database, session manager, automatic memory, or automatic planning.

The [request-context example](https://github.com/DeanDiasti/seqoria_agent/blob/main/examples/conversation/request_context.py) demonstrates independent histories and identities without a hosted model.

## Application lifespan

This FastAPI lifespan creates reusable provider connections, a fixed tool catalogue, and a shared tool concurrency budget. Install the `server` extra to use FastAPI.

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.providers import AsyncOpenAIProvider

tools = LocalToolbox()


@tools.tool(server_populated=["tenant_id", "user_id"])
async def account_summary(tenant_id: str, user_id: str) -> dict:
    """Read the authenticated user's account summary."""
    # Replace account_service with your application's authorized data service.
    return await account_service.read(tenant_id=tenant_id, user_id=user_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncOpenAIProvider(model_id=os.environ["OPENAI_MODEL"]) as provider:
        async with AsyncAIClient(
            providers=[provider],
            local_toolbox=tools,
            max_concurrency=16,
            tool_timeout=30,
        ) as client:
            app.state.agent = client
            yield


app = FastAPI(lifespan=lifespan)
```

The client owns its MCP transports and default transport HTTP pool. Providers belong to the application; close them after their clients. Create resources inside each worker's running event loop, rather than as global async singletons or before forking.

## Request handling

After authenticating the request, authorizing conversation ownership, and acquiring your application lock for that conversation:

```python
# history: loaded from your authorized conversation storage
# principal: established by your authentication layer
async for event in app.state.agent.chat(
    user_text,
    history=history,
    context={
        "tenant_id": principal.tenant_id,
        "user_id": principal.user_id,
    },
):
    # Translate message events to your HTTP or WebSocket response.
    ...

serialized_history = [
    message if isinstance(message, dict) else message.model_dump(mode="json") for message in history
]
# Persist serialized_history before releasing the conversation lock.
```

`account_service`, `principal`, history loading, persistence, and locking are application integration points. Never copy identity from the model or an unverified request field into trusted context.

One client can serve independent conversations concurrently. A history list is mutable and must have one writer at a time; concurrent turns in the same conversation need serialization. Multiple workers require storage transactions or distributed locks, rather than process-local locks alone. Add a conversation version check when saving to prevent lost updates.

Own context budgeting and retention in your application. Preserve tool calls and results together when pruning history. On a cancelled or failed turn, reconcile incomplete history before retrying; tool side effects might already have happened. Persist critical operations independently and use idempotency keys where replay is possible.

## Boundaries

- Tool concurrency limits are per client, per worker. Add application-level request limits for model calls, payload sizes, total request time, and queue depth.
- Use async tools for cancellable network and database I/O. Sync tools run in worker threads and continue running if their caller times out; CPU-heavy operations need separate workers.
- Configure a shared client's providers, credentials, remote servers, and blocklist for the whole lifespan. Do not change these to implement individual users' permissions.
- Remote MCP sessions have their own authentication state. Use separate client/session lifespans when connecting as different end users. Remote servers resolve hidden identity parameters from their authenticated server context.
- Tool results are not cached by the library. If caching is needed, implement it in the application operation or data service, with explicit ownership, invalidation, authorization scope, and replay rules.
- Dependency declarations schedule tools requested in the same batch; they do not guarantee that an authentication tool ran or enforce access control. Authorize inside the operation itself.

This release includes concurrency and transport regression tests. It does not claim measured throughput for any hosted model, database, or deployment size; benchmark the actual application workload before sizing workers.
