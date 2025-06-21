# New test file for strict-mode validation

import xml.etree.ElementTree as ET

import pytest

from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.server import XMCP


def _error_code(xml_str: str) -> str:
    root = ET.fromstring(xml_str)
    err = root.find("error")
    return err.attrib["code"] if err is not None else ""


def test_display_envelope_passes_strict_mode():
    mcp = XMCP(name="strict", strict_mode=True)

    def ok_tool():  # noqa: D401
        return DisplayEnvelope(type="markdown", payload="hi")

    # Manually wrap with the XMCP error/validation handler
    wrapped = mcp._wrap_tool_with_error_handling(func=ok_tool, tool_name="ok_tool", formatter_cls=None)
    xml_payload = wrapped()

    root = ET.fromstring(xml_payload)
    assert root.find("display") is not None


def test_invalid_payload_yields_error():
    mcp = XMCP(name="strict", strict_mode=True)

    def bad_tool():  # noqa: D401
        return {"foo": "bar"}  # Not a valid DisplayEnvelope

    wrapped = mcp._wrap_tool_with_error_handling(bad_tool, "bad_tool")
    xml = wrapped()
    assert _error_code(xml) == "INVALID_PAYLOAD"


def test_malformed_xml_yields_error():
    mcp = XMCP(name="strict", strict_mode=True)

    def xml_tool():  # noqa: D401
        return "<tool_response tool_name=\"x\"><invalid></tool_response>"

    wrapped = mcp._wrap_tool_with_error_handling(xml_tool, "xml_tool")
    xml_payload = wrapped()
    assert _error_code(xml_payload) == "INVALID_XML" 