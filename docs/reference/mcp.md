# MCP transport API

Most applications configure `MCPServerConfig` values on `AsyncAIClient`. Read [MCP connections](../guides/mcp.md) before using these lower-level APIs. OAuth is supported only with Streamable HTTP. Authentication and hidden identity resolution for remote tools belong to their server.

## Server and OAuth configuration

::: seqoria_agent.client.mcp_transport.MCPServerConfig

::: seqoria_agent.client.mcp_transport.OAuthConfig

::: seqoria_agent.client.mcp_transport.InMemoryTokenStorage

In-memory OAuth storage is process-local and temporary. Production token persistence, encryption, tenancy, consent, and refresh behavior need application-specific review.

## Connection ownership

Use the factory or convenience context manager for a connection. The returned `MCPConnection` is valid only inside that context. A supplied HTTP pool is caller-owned; an internally created pool is closed on exit. Connection pools own their sessions and should be entered and closed within one application event loop.

::: seqoria_agent.client.mcp_transport.connect_mcp

::: seqoria_agent.client.mcp_transport.MCPTransportFactory

::: seqoria_agent.client.mcp_transport.MCPConnection

::: seqoria_agent.client.mcp_transport.HTTPClientPool

::: seqoria_agent.client.ai_client.connection_pool.MCPConnectionPool
