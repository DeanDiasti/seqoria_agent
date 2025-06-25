# Authentication Guide

This page covers how to **secure** your Seqoria-Agent applications using the new, optional authentication framework shipped in `v0.4+`.

> Using auth is *entirely optional*.  If you do nothing, the previous no-auth behaviour remains unchanged.

---

## 1. Terminology & Building Blocks

| Layer | Class | Purpose |
|-------|-------|---------|
| **Server** | `BaseAuthBackend` | Validate **incoming** requests and, on success, inject a *principal* object on `request.state.principal`. |
| | `BaseMiddlewareAuthBackend` | Same as above **and** a Starlette `BaseHTTPMiddleware` subclass so it can be mounted via `app.add_middleware(...)`. |
| **Client** | `BaseAuthProvider` | Attach **outgoing** headers / cookies to every HTTP request issued by `AIClient`. |

Concrete helpers shipped with the library:

### Back-ends (server)
* `NoAuthBackend` – default, always succeeds.
* `BearerCookieBackend` – validates a JWT carried in a cookie.
* `HeaderApiKeyBackend` – checks a static API-key in a custom HTTP header.

### Providers (client)
* `NoAuthProvider` – sends no auth data (default).
* `StaticCookieProvider` – adds a static cookie (JWT) on every request.
* `StaticBearerTokenProvider` – adds `Authorization: Bearer <token>`.
* `StaticApiKeyProvider` – adds a custom API-key header.

---

## 2. Securing a FastXMCP Server

### 2.1 Constructor wiring (simple)
```python
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import BearerCookieBackend

app = FastXMCP(
    auth_backend=BearerCookieBackend(secret=JWT_SECRET, algorithm="HS256")
)
```
`FastXMCPRouter` and `LocalToolbox` accept the **same** `auth_backend` kwarg.

### 2.2 Middleware wiring (advanced)
Because every backend now subclasses `BaseMiddlewareAuthBackend`, you can also mount it on **any FastAPI app**:

```python
from fastapi import FastAPI
from seqoria_agent.auth.backends import HeaderApiKeyBackend

app = FastAPI()
app.add_middleware(HeaderApiKeyBackend, key_header="X-API-Key", validator="demo")
```

If this app later includes a `FastXMCP` router, the middleware protects *both* classic REST routes **and** XMCP tool calls.

### 2.3 Server-populated parameters
Mark tool parameters that should be injected by the server:

```python
@app.tool(server_populated=["user_id"])
async def get_profile(user_id: str):
    ...
```
When a request succeeds, `user_id` is automatically populated from `principal.id` (dictionary keys work too).  The LLM cannot override the value.

Need the whole principal?  Use `server_populated=["principal"]` instead.

---

## 3. Authenticating an AIClient

Pass an **AuthProvider** when constructing the client:

```python
from seqoria_agent import AIClient
from seqoria_agent.auth.providers import StaticApiKeyProvider
from seqoria_agent.providers import OpenAIProvider

async with AIClient(
    urls=["https://api.mytools.ai/mcp/"],
    providers=[OpenAIProvider(model_id="gpt-4o")],
    auth_provider=StaticApiKeyProvider(token="demo", header_name="X-API-Key"),
) as ai:
    res = await ai.chat("ping", history=[])
    print(res["answer"])
```

The provider can **refresh** tokens lazily by overriding `refresh_token_if_needed()`.

`AIClient.add_auth_provider(new_provider)` lets you hot-swap credentials at runtime without recreating the client.

---

## 4. Using the back-end as a FastAPI Dependency

Every backend offers a zero-boilerplate helper:

```python
backend = BearerCookieBackend(secret=JWT_SECRET)

@app.get("/me")
async def me(user = backend.as_dependency()):
    return {"id": user.get("id")}
```

The same instance can protect XMCP **and** regular routes.

---

## 5. End-to-end Example (API-Key)
See `examples/auth/api_key_middleware_server.py` for a runnable demo:

```bash
uvicorn examples.auth.api_key_middleware_server:app
```
Then query it:
```bash
curl -H "X-API-Key: demo" http://localhost:8000/health
```

---

## 6. Migration Checklist
* **Doing nothing?**  Everything keeps working.
* Swap custom dep-based auth with a single backend + `server_populated=[...]`.
* Update clients to pass a matching provider.

---

## 7. Reference
| Symbol | Module | Notes |
|--------|--------|-------|
| `BaseAuthBackend` | `seqoria_agent.auth.backends` | Core ABC |
| `BaseMiddlewareAuthBackend` | same | Inherits `BaseHTTPMiddleware` |
| `BearerCookieBackend` | same | JWT in cookie |
| `HeaderApiKeyBackend` | same | Static API-key |
| `BaseAuthProvider` | `seqoria_agent.auth.providers` | Client ABC |
| `StaticCookieProvider` | same | Static cookie |
| `StaticBearerTokenProvider` | same | `Authorization` header |
| `StaticApiKeyProvider` | same | Custom header |

