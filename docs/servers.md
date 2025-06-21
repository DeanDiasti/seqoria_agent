# Building an XMCP Server

`XMCP` is a thin wrapper around FastAPI + Model Context Protocol (MCP) that adds:

* `@display_tool` decorator for rich UI payloads
* Automatic **XML envelopes** for deterministic prompt parsing
* **Structured error channel** via `<error>` tags
* Declarative **dependency graph** with `depends_on` annotation
* **Server-populated parameters** to inject user_id, geo, etc.

---

## 1. Regular vs Display Tools

```python
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.models.envelope import DisplayEnvelope
mcp = FastXMCP()

@mcp.tool
def add(a: int, b: int) -> int:
    return a + b

@mcp.display_tool
def greet(name: str, show_user: bool = False):
    if show_user:
        return DisplayEnvelope(type="markdown", payload=f"### Hello {name}! 👋")
    return {"greeting": f"Hello {name}!"}
```

## 2. Dependencies

```python
@mcp.tool
def current_time() -> str: ...

@mcp.tool(depends_on=["current_time"])
def timestamped_divide(a: float, b: float):
    return f"{current_time()} – {a/b}"  # safe: current_time ran first
```

When the LLM requests both tools in the same turn, `timestamped_divide` waits for `current_time`.

## 3. Strict Mode

Enable at app level to validate every payload.

```python
mcp = XMCP(strict_mode=True)
```

Invalid XML or non-DisplayEnvelope returns turn into a structured error with `code="INVALID_PAYLOAD"`.

## 4. Server-Populated Parameters (SPP)

```python
@mcp.tool(server_populated=["user_id"])
def get_profile(user_id: str, name: str):
    ...

@mcp.register_param_provider
async def _hdr(req, tool_name, param_name):
    if param_name == "user_id":
        return req.headers.get("X-User-Id")
```

The LLM **cannot** override `user_id` unless you set `strict_server_params=False`.

## 5. Running the Server

```bash
uvicorn my_server:mcp --port 3333
```

Use `--reload` for hot-reloading during development. 