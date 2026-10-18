"""Client-side SDK for MCP-based AI applications.

This module exports:
- AIClient: Synchronous client for blocking usage
- AsyncAIClient: Asynchronous client for async/await usage
- MCPTransportFactory: Factory for MCP transport connections
- MCPServerConfig: Configuration for MCP servers
- MCPConnectionPool: Connection pool with resilience features
- OAuthConfig: Configuration for OAuth 2.1 authentication
- InMemoryTokenStorage: In-memory OAuth token storage
"""

from seqoria_agent.client.ai_client import AIClient, AsyncAIClient
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.mcp_transport import (
    InMemoryTokenStorage,
    MCPServerConfig,
    MCPTransportFactory,
    OAuthConfig,
)

__all__ = [
    "AIClient",
    "AsyncAIClient",
    "MCPServerConfig",
    "MCPTransportFactory",
    "MCPConnectionPool",
    "OAuthConfig",
    "InMemoryTokenStorage",
]
