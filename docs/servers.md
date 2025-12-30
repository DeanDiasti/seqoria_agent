# Building a FastXMCP Server

`XMCP` is a thin wrapper around FastAPI + Model Context Protocol (MCP) that adds:

* `@display_tool` decorator for rich UI payloads
* Automatic **XML envelopes** (optionally wrapped in JSON) for deterministic prompt parsing
* Declarative **XML formatting** with automatic schema derivation
* **Structured error channel** via `<error>` tags
* Declarative **dependency graph** with `depends_on` annotation
* **Server-populated parameters** to inject user_id, geo, etc.
* **Modular routers** with `FastXMCPRouter` for large projects

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

### JSON Wrapper & Declarative Formatter (v0.3+)

Enable the JSON envelope and influence the **inner** XML tags with a declarative formatter:

```python
from seqoria_agent.server.formatters import make_formatter

app = FastXMCP(json_wrapper=True)  # ← responses are now {"ai_message": xml, "metadata": {...}}

# Build a tiny formatter inline – emits <product> inside <tool_response>
product_fmt = make_formatter(top_level_fields=["product"])

@app.tool(response_format=product_fmt)
def multiply(a: int, b: int):
    return {"product": int(a) * int(b)}

# No formatter supplied?  FastXMCP derives one automatically from the *first* result
@app.tool()
def add(a: int, b: int):
    return int(a) + int(b)  # → <sum>7</sum> (derived on first call)
```

`json_wrapper=False` by default for backward-compatibility.

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
app = FastXMCP(strict_mode=True)
```

Invalid XML or non-DisplayEnvelope returns turn into a structured error with `code="INVALID_PAYLOAD"`.

## 4. Server-Populated Parameters (SPP)

```python
@mcp.tool(server_populated=["user_id"])
def get_profile(user_id: str, name: str):
    ...

```

The LLM **cannot** override `user_id` unless you set `strict_server_params=False`.

## 5. Running the Server

```bash
uvicorn my_server:app --port 3333
```

Use `--reload` for hot-reloading during development.

## 6. Modular Routers

Use `FastXMCPRouter` (similar to FastAPI's `APIRouter`) to group related tools
and mount them on a main application.  All router tools are automatically
merged into the parent registry and exposed via `tools/list` & `tools/call`.

```python
from seqoria_agent.server.fastxmcp import FastXMCP, FastXMCPRouter

math = FastXMCPRouter(prefix="/math", tags=["math"])

@math.tool()
def add(a: int, b: int) -> int:
    return a + b

@math.tool()
def multiply(a: int, b: int) -> int:
    return a * b

app = FastXMCP(title="Utility Server")
app.include_router(math)

if __name__ == "__main__":
    app.run(port=8000)
```

If two routers (or the parent app) attempt to register the **same tool name**
`include_router()` raises `XMCPError(code="DUPLICATE_TOOL")` to prevent
ambiguous calls. 

## 7. Securing Tools

Authentication helpers were removed from this library. Secure your MCP endpoint using your existing infrastructure:

- Run behind a reverse proxy / API gateway (recommended).
- Use FastAPI middleware in your own app to validate headers/cookies before requests reach the MCP server.