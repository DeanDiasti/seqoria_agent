# Authentication Guide

Seqoria-Agent ships with a *pluggable* authentication layer that works the same for
local `LocalToolbox`, in-process `FastXMCP` apps and remote HTTP servers.

## Quick start – Cookie based JWT

Server:
```python
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import BearerCookieBackend

app = FastXMCP(
    auth_backend=BearerCookieBackend(secret="my-secret", algorithm="HS256")
)

@app.tool(server_populated=["user_id"])
async def hello(user_id: str):
    return {"msg": f"hello {user_id}"}
```

Client:
```python
from seqoria_agent.client import AIClient
from seqoria_agent.auth.providers import StaticCookieProvider
from seqoria_agent.providers.mock import DummyLLM

token = "<jwt-from-your-auth-system>"
async with AIClient(
    providers=[DummyLLM()],
    urls=["http://localhost:8000/mcp/"],
    auth_provider=StaticCookieProvider(token),
) as c:
    print(await c.call_mcp_tool("hello", {}))
```

## Built-in helpers

* **Backends**
  * `NoAuthBackend` – default, allows everything.
  * `BearerCookieBackend` – validates JWT stored in a cookie.
  * `HeaderApiKeyBackend` – checks static API-key in a HTTP header.
* **Providers**
  * `NoAuthProvider` – default, sends no credentials.
  * `StaticCookieProvider` – static cookie header.
  * `StaticBearerTokenProvider` – `Authorization: Bearer <token>` header.
  * `StaticApiKeyProvider` – `X-API-Key: <token>` header (configurable name).

Backends and providers are plain Python classes, so you can subclass them to
integrate OAuth2 introspection, API keys or mTLS.

## Per-router security

Routers can specify their own backend:
```python
from seqoria_agent.server.fastxmcp.router import FastXMCPRouter
from seqoria_agent.auth.backends import BearerCookieBackend

secure = FastXMCPRouter(auth_backend=BearerCookieBackend(secret="s", algorithm="HS256"))
```
Tools registered on that router inherit the stricter backend while the rest of
your app can stay unauthenticated.

## Token refresh hook

If your access token can expire, implement `refresh_token_if_needed` in your
provider. The HTTP client calls it before *every* request.

---
For the full design rationale see `oauth-implementation.md`. 