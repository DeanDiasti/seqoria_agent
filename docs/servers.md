# MCP servers

`SeqoriaMCP` extends the official MCP Python SDK's `FastMCP` server. Its transport, authentication, and deployment settings come from that SDK.

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

Connect a client with `mcp_servers=["http://127.0.0.1:8000/mcp"]`. A bare HTTP origin is expanded to `/mcp`; explicit endpoint paths are preserved. Use `MCPServerConfig` for headers, timeout, transport type, subprocess environment, or OAuth. OAuth's default interactive browser/loopback flow is intended for local scripts; a web service must provide its own user authorization flow and handlers.

## Hidden server parameters

```python
@server.register_param_provider
async def resolve_identity(ctx, tool_name, param_name):
    # Authenticate the transport and verify authorization in your application.
    if param_name == "tenant_id":
        return await authenticated_tenant_for_request(ctx)
    return None

@server.tool(server_populated=["tenant_id"])
async def report(region: str, tenant_id: str) -> dict:
    """Read a regional report for the authenticated tenant."""
    return await reports.read(tenant_id=tenant_id, region=region)
```

Hidden parameters are omitted from the public tool schema. Their values come from server-side providers or authenticated request context on every call; missing values fail closed. Do not accept caller-supplied tenant or user IDs as proof of identity. The client does not transmit local `server_args` to remote tools.

Dependency annotations apply to requested calls in the agent's executor. They are scheduling hints, not a security boundary. Validate access inside each tool even when another tool declares an authorization dependency.

Synchronous functions execute in worker threads. `format_tool_response()` and `format_error_response()` remain available for applications that need explicit response envelopes; ordinary tools can return normal Python values.
