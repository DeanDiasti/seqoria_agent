"""Tests for SeqoriaMCP tool registration and routing.

Note: SeqoriaMCP uses MCP SDK's FastMCP which has built-in tool routing.
These tests verify our extended features like server-populated parameters.
"""

import pytest

from seqoria_agent.server import SeqoriaMCP


@pytest.fixture()
def mcp_server():
    """Create a SeqoriaMCP server with test tools."""
    mcp = SeqoriaMCP("TestServer")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        return int(a) + int(b)

    @mcp.tool(server_populated=["secret"])
    def reveal_secret(name: str, secret: str) -> str:
        return f"Hello {name}, secret={secret}"

    return mcp


def test_tool_metadata_registered(mcp_server):
    """Verify tool metadata is registered correctly."""
    metadata = mcp_server.tool_metadata

    assert "add" in metadata
    assert metadata["add"].server_populated == []

    assert "reveal_secret" in metadata
    assert metadata["reveal_secret"].server_populated == ["secret"]


def test_get_tool_annotations(mcp_server):
    """Verify annotations can be retrieved."""
    ann = mcp_server.get_tool_annotations("reveal_secret")
    assert "server_populated" in ann
    assert "secret" in ann["server_populated"]


def test_format_tool_response(mcp_server):
    """Verify response formatting."""
    result = mcp_server.format_tool_response("add", 42)
    assert "42" in result


def test_format_tool_response_xml(mcp_server):
    """Verify XML response formatting."""
    result = mcp_server.format_tool_response("add", {"value": 42}, as_xml=True)
    assert "<tool_response" in result
    assert 'tool_name="add"' in result
    assert "42" in result


def test_format_error_response(mcp_server):
    """Verify error response formatting."""
    result = mcp_server.format_error_response(
        "add",
        "VALIDATION_ERROR",
        "Invalid argument",
        retryable=True,
    )
    assert "<tool_response" in result
    assert "VALIDATION_ERROR" in result
    assert "Invalid argument" in result
    assert 'retryable="true"' in result


def test_tool_with_dependencies():
    """Verify tools can declare dependencies."""
    mcp = SeqoriaMCP("DepTest")

    @mcp.tool()
    def step1() -> str:
        return "done"

    @mcp.tool(depends_on=["step1"])
    def step2() -> str:
        return "also done"

    meta = mcp.tool_metadata["step2"]
    assert "step1" in meta.depends_on
