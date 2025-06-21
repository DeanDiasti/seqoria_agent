# Server Recipes

This page shows cookbook-style snippets for building XMCP servers on top of the **Model Context Protocol (MCP)** runtime.

> **TL;DR** – import `XMCP`, decorate your functions, and call `.run()`.

## 1. Regular Tool

```python
from xmcp import XMCP
mcp = XMCP()

@mcp.tool
def add(a: int, b: int) -> int:
    return a + b
```

```
uvicorn my_server:mcp --port 3333
```

## 2. Display Tool

```python
from xmcp import XMCP, DisplayEnvelope
mcp = XMCP()

@mcp.display_tool
def sales_summary(region: str, show_user: bool = False):
    data = {"region": region, "sales": 42}
    if show_user:
        return DisplayEnvelope(type="table", payload=[data], title=f"Sales {region}")
    return data
```

## 3. Custom XML Formatter

```python
from xmcp.server.formatters import DeclarativeXMLFormatter

class WeatherFormatter(DeclarativeXMLFormatter):
    top_level_fields = ["city", "temperature", "condition"]

@mcp.tool(xml_formatter=WeatherFormatter)
def get_weather(city: str):
    return {"city": city, "temperature": 18, "condition": "Sunny", "humidity": 55}
```

The LLM sees only the whitelisted fields, keeping the prompt tight.

## 4. Structured Errors

```python
from xmcp.models.errors import XMCPError

@mcp.tool
def divide(a: float, b: float):
    if b == 0:
        raise XMCPError(code="DIV_BY_ZERO", detail="Cannot divide by zero", retryable=False)
    return a / b
```

`AIClient` will receive an `<error>` envelope and can retry or surface a nicer message.

## 5. Server-Populated Parameters

```python
@mcp.tool(server_populated=["user_id"])
def greet(user_id: str, name: str):
    return f"Hello {name} (uid={user_id})"
```

Register a provider:

```python
@ mcp.register_param_provider
aSync def user_id_from_header(request, tool_name, param_name):
    if param_name == "user_id":
        return request.headers.get("X-User-Id")
```

## 6. Dependency Graph

```python
@mcp.tool
def current_time() -> str: ...

@mcp.tool(depends_on=["current_time"])
def timestamped_divide(a: float, b: float) -> str: ...
```

If both tools are requested in the same turn, `current_time` will run first. 