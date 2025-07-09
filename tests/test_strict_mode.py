# New test file for strict-mode validation

import xml.etree.ElementTree as ET

from seqoria_agent.server.fastxmcp import FastXMCP


def _error_code(xml_str: str) -> str:
    root = ET.fromstring(xml_str)
    err = root.find("error")
    return err.attrib["code"] if err is not None else ""


def test_invalid_payload_yields_error():
    mcp = FastXMCP(name="strict", strict_mode=True)

    def bad_tool():  # noqa: D401
        return 123  # Invalid payload type (neither dict/list nor XML)

    wrapped = mcp._wrap_tool_with_error_handling(bad_tool, "bad_tool")
    xml = wrapped()
    assert _error_code(xml) == "INVALID_PAYLOAD"


def test_malformed_xml_yields_error():
    mcp = FastXMCP(name="strict", strict_mode=True)

    def xml_tool():  # noqa: D401
        return "<tool_response tool_name=\"x\"><invalid></tool_response>"

    wrapped = mcp._wrap_tool_with_error_handling(xml_tool, "xml_tool")
    xml_payload = wrapped()
    assert _error_code(xml_payload) == "INVALID_XML" 