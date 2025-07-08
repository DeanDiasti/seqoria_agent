# Multi-Tenancy & Server-Populated Parameters

Seqoria-Agent can operate in true **multi-tenant** mode – one running service
instance serves traffic for many tenants while keeping their data and
cache-entries isolated.

At runtime the tenant is identified through request metadata (e.g. a signed
JWT, sub-domain or plain HTTP header).  The backend injects this information
into **server-populated parameters (SPP)** such as `tenant_id` or `user_id`.

With SPP **v2** the client uses a simple `ParameterRegistry`:

```python
from seqoria_agent.client import AIClient
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.providers import OpenAIProvider

registry = ParameterRegistry({
    # constant – forwarded as-is
    "tenant_id": "acme-corp",
    # callable – receives a ParamContext and may be async
    "user_id": lambda tenant_id: f"{tenant_id}:anonymous",
})

async with AIClient(
    providers=[OpenAIProvider(model_id="gpt-4o")],
    urls=["https://mcp.example.com/mcp"],
    param_registry=registry,
) as client:
    # the LLM never sees tenant_id/user_id – they’re injected on the wire
    await client.call_mcp_tool("hello", {})
```

When a **callable** is supplied the registry passes only the *keyword
arguments* that match its signature – no brittle positional contract:

```python
from typing import TypedDict

class ParamContext(TypedDict, total=False):
    tenant_id: str
    user_id: str
    history: list[dict]
    request_meta: dict
```

Back-end side tools declare required SPP via the `server_populated` annotation:

```python
@app.tool(server_populated=["user_id", "tenant_id"])
async def hello(user_id: str, tenant_id: str):
    return {"msg": f"hi {user_id} from {tenant_id}"}
```

## Cache isolation

Functions like `build_cache_key` automatically include `tenant_id` so cached
results never leak across tenants. 