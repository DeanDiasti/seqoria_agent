"""Tests for SeqoriaMCP response formatting and error handling."""

import pytest

from seqoria_agent.server import SeqoriaMCP


def test_format_tool_response_string():
    """String results are returned as-is."""
    mcp = SeqoriaMCP("test")
    result = mcp.format_tool_response("my_tool", "Hello world")
    assert result == "Hello world"


def test_format_tool_response_dict():
    """Dict results are JSON-serialized."""
    mcp = SeqoriaMCP("test")
    result = mcp.format_tool_response("my_tool", {"key": "value", "num": 42})
    assert '"key": "value"' in result
    assert '"num": 42' in result


def test_format_tool_response_xml():
    """XML format wraps content in envelope."""
    mcp = SeqoriaMCP("test")
    result = mcp.format_tool_response("my_tool", "test data", as_xml=True)
    
    assert "<tool_response" in result
    assert 'tool_name="my_tool"' in result
    assert "<llm_output>" in result
    assert "test data" in result


def test_format_error_response():
    """Error responses include code and detail."""
    mcp = SeqoriaMCP("test")
    result = mcp.format_error_response(
        "bad_tool",
        "VALIDATION_ERROR",
        "Invalid input: x must be positive",
        retryable=False,
    )

    assert "<tool_response" in result
    assert 'tool_name="bad_tool"' in result
    assert "VALIDATION_ERROR" in result
    assert "Invalid input" in result
    assert 'retryable="false"' in result


def test_format_error_response_retryable():
    """Retryable errors are marked correctly."""
    mcp = SeqoriaMCP("test")
    result = mcp.format_error_response(
        "flaky_tool",
        "TIMEOUT",
        "Service temporarily unavailable",
        retryable=True,
    )

    assert 'retryable="true"' in result


def test_format_error_response_with_meta():
    """Error responses can include metadata."""
    mcp = SeqoriaMCP("test")
    result = mcp.format_error_response(
        "api_tool",
        "RATE_LIMITED",
        "Too many requests",
        retryable=True,
        meta={"retry_after": 30},
    )

    assert "<meta>" in result
    assert "retry_after" in result


def test_tool_metadata_description():
    """Tool metadata includes description from docstring."""
    mcp = SeqoriaMCP("test")

    @mcp.tool()
    def my_tool(x: int) -> int:
        """Add one to the input."""
        return x + 1

    meta = mcp.tool_metadata.get("my_tool")
    assert meta is not None
    assert "Add one" in meta.description
