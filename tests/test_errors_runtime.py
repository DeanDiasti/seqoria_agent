"""Tests for structured error handling in `xmcp.server.runtime.XMCP`."""

import xml.etree.ElementTree as ET

import pytest

from seqoria_agent.server import XMCP
from seqoria_agent.models.errors import XMCPError


def _extract_error_detail(xml_str: str):
    root = ET.fromstring(xml_str)
    err = root.find("error")
    return err.attrib["code"], err.attrib["retryable"], (err.text or "").strip()


def test_build_error_xml_sync():
    mcp = XMCP(name="test")

    def failing_tool():
        raise XMCPError(code="FOO", detail="bar", retryable=True)

    wrapper = mcp._wrap_tool_with_error_handling(failing_tool, "failing_tool")
    xml_payload = wrapper()

    code, retryable, detail = _extract_error_detail(xml_payload)
    assert code == "FOO"
    assert retryable == "true"
    assert detail == "bar"


test_payload = XMCPError(code="ERR", detail="bad", retryable=False)


@pytest.mark.asyncio
async def test_build_error_xml_async():
    mcp = XMCP(name="test_async")

    async def async_fail():  # noqa: D401
        raise XMCPError(code="ERR", detail="bad", retryable=False)

    wrapper = mcp._wrap_tool_with_error_handling(async_fail, "async_fail")
    xml_payload = await wrapper()

    code, retryable, detail = _extract_error_detail(xml_payload)
    assert code == "ERR"
    assert retryable == "false"
    assert detail == "bad" 