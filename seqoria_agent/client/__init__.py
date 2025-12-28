"""Client-side SDK for MCP-based AI applications.

This module exports:
- AIClient: Synchronous client for blocking usage
- AsyncAIClient: Asynchronous client for async/await usage
- MCPTransportFactory: Factory for MCP transport connections
- MCPServerConfig: Configuration for MCP servers
- MCPConnectionPool: Connection pool with resilience features
"""

from seqoria_agent.client.ai_client import AIClient, AsyncAIClient
from seqoria_agent.client.mcp_transport import (
    MCPServerConfig,
    MCPTransportFactory,
)
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool

__all__ = [
    "AIClient",
    "AsyncAIClient",
    "MCPServerConfig",
    "MCPTransportFactory",
    "MCPConnectionPool",
]
