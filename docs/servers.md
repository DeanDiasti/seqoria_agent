# Building an XMCP Server

The `mcpx.XMCP` class is the heart of your tool server. It's a specialized subclass of `fastmcp.FastMCP` that adds support for display-oriented tools and structured data formats.

## Basic Usage

You create an `XMCP` instance and decorate your Python functions to expose them as tools.

```python
from mcpx import XMCP

mcp = XMCP()

@mcp.tool
def my_regular_tool(param: str) -> dict:
    """This tool returns data for the LLM to process."""
    return {"result": f"You said {param}"}
```

## Display Tools

For tools that can generate rich UI outputs, use the `@mcp.display_tool` decorator. This allows the tool to return a `DisplayEnvelope` that the `AIClient` can use to render UI components, bypassing a follow-up LLM call.

The decision is controlled by the `show_user: bool` parameter, which is automatically injected into your function's signature if you include it.

```python
from mcpx import XMCP, DisplayEnvelope

mcp = XMCP()

@mcp.display_tool
def get_user_profile(user_id: str, show_user: bool = False):
    """
    Fetches a user profile.
    
    - If `show_user` is `False`, returns a dict for the LLM.
    - If `show_user` is `True`, returns a `DisplayEnvelope` for the UI.
    """
    profile_data = {"id": user_id, "name": "Alex Doe", "email": "alex@example.com"}
    
    if show_user:
        return DisplayEnvelope(
            type="markdown",
            title=f"User Profile: {profile_data['name']}",
            payload=f"**ID:** {user_id}\n\n**Email:** {profile_data['email']}"
        )
    
    return profile_data
```

## Structured Errors

`XMCP` automatically catches `XMCPError` exceptions and converts them into a structured XML response that the `AIClient` and LLM can understand, enabling graceful failure handling.

```python
from mcpx.models import XMCPError

@mcp.tool
def get_sensitive_data(user_id: str):
    if not user_id.startswith("admin-"):
        raise XMCPError(
            code="UNAUTHORIZED",
            detail="Access denied for non-admin users.",
            retryable=False
        )
    # ... return data
```

## Custom XML Formatting

For regular `@tool`s, you can control the XML representation of the output that the LLM sees by providing an `xml_formatter`.

See the API reference for `mcpx.server.formatters` for more details.

```python
from mcpx.server.formatters import DeclarativeXMLFormatter

class SimpleFormatter(DeclarativeXMLFormatter):
    top_level_fields = ["id", "name"]

@mcp.tool(xml_formatter=SimpleFormatter)
def get_internal_data(item_id: str):
    return {"id": item_id, "name": "Project X", "internal_code": "PX-123"}
```

The LLM will only see the `id` and `name` fields, keeping the context clean and focused. 