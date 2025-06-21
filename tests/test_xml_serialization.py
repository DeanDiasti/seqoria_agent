"""Tests for XMCP XML success serialization via custom formatters and DisplayEnvelope."""

import xml.etree.ElementTree as ET

import pytest

from seqoria_agent.server import XMCP
from seqoria_agent.server.formatters import DeclarativeXMLFormatter
from seqoria_agent.models.envelope import DisplayEnvelope


class _KeepFormatter(DeclarativeXMLFormatter):
    """Only include the *keep* field from the top-level result."""

    top_level_fields = ["keep"]


# Helper simply checks substring inside full XML because formatted content is wrapped in CDATA


def _tag_present(xml_str: str, tag: str) -> bool:  # noqa: D401 – helper
    return f"<{tag}>" in xml_str


def test_formatter_whitelist():
    mcp = XMCP(name="xml_test")

    def data_tool():  # noqa: D401
        return {"keep": 1, "drop": 2}

    wrapper = mcp._wrap_tool_with_error_handling(
        func=data_tool, tool_name="data_tool", formatter_cls=_KeepFormatter
    )
    xml_payload = wrapper()

    # The <keep> tag should be present; <drop> should be filtered out
    assert _tag_present(xml_payload, "keep")
    assert not _tag_present(xml_payload, "drop")


def test_display_envelope_serialization():
    mcp = XMCP(name="display")

    def disp_tool():  # noqa: D401
        return DisplayEnvelope(type="notification", payload="hi")

    wrapper = mcp._wrap_tool_with_error_handling(
        func=disp_tool, tool_name="disp_tool", formatter_cls=None
    )
    xml_payload = wrapper()

    root = ET.fromstring(xml_payload)
    display_node = root.find("display")
    assert display_node is not None
    # Ensure llm_output absent because DisplayEnvelope path should not emit it
    assert root.find("llm_output") is None 