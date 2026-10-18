# MCP client connections

Seqoria Agent connects to independently hosted MCP servers. It supports Streamable HTTP, SSE, and stdio; authoring and deploying those servers belongs to their server implementation.

Configure remote servers before entering the client lifespan:

```python
from seqoria_agent.client import MCPServerConfig

server = MCPServerConfig(
    url="http://127.0.0.1:8000/mcp",
    timeout=30,
)
# Pass mcp_servers=[server] to AsyncAIClient along with your providers.
```

A bare HTTP origin is expanded to `/mcp`; explicit endpoint paths are preserved. `MCPServerConfig` also accepts headers, transport selection, subprocess environment, and OAuth configuration. Enter `async with client:` before listing or calling remote tools. The client owns its sessions and closes them on exit.

OAuth is supported for Streamable HTTP only; configuring it on SSE or stdio raises an error. The default interactive OAuth browser/loopback flow is for local scripts. A web service must supply its own user authorization flow and handlers. Remote MCP servers must authenticate callers, enforce authorization, and resolve any hidden identity parameters themselves. Local `server_args` are not transmitted to remote tools. Use separate client/session lifespans for remote connections authenticated as different end users.

For in-process Python functions, use `LocalToolbox`. For remote hosting, use the official MCP SDK or another compatible server implementation. The [stdio example](https://github.com/DeanDiasti/seqoria_agent/blob/main/examples/mcp/stdio_client.py) starts and closes its own local SDK server subprocess without a model API key. The [HTTP client example](https://github.com/DeanDiasti/seqoria_agent/blob/main/examples/mcp/client.py) connects to the independently started example servers; see the [catalog](https://github.com/DeanDiasti/seqoria_agent/blob/main/examples/README.md) for commands.

Dependency annotations apply to requested calls in the client's executor. They schedule calls rather than enforce authorization. Each remote tool must validate access to its own data and operations.
