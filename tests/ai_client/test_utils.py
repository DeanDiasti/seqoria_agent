import pytest

from seqoria_agent.client.ai_client.utils import extract_error_from_xml, build_cache_key, null_async_cm


class TestExtractErrorFromXML:
    """Focused tests for the XML error helper."""

    def test_detects_and_parses_error_block(self):  # noqa: D401
        xml = (
            """
            <tool_response>
                <error code="FOO" retryable="true">
                    <![CDATA[Something went wrong]]>
                </error>
            </tool_response>
            """
        )
        result = extract_error_from_xml(xml)
        assert result == {
            "code": "FOO",
            "retryable": True,
            "detail": "Something went wrong",
            "meta": None,
        }

    def test_returns_none_when_no_error_tag(self):  # noqa: D401
        xml = "<tool_response><result>ok</result></tool_response>"
        assert extract_error_from_xml(xml) is None

    def test_meta_parsed_when_json_present(self):  # noqa: D401
        xml = (
            """
            <wrapper>
              <error code="BAR" retryable="false">
                <meta>{\"foo\": 123}</meta>
              </error>
            </wrapper>
            """
        )
        result = extract_error_from_xml(xml)
        assert result is not None and result["meta"] == {"foo": 123}


def test_build_cache_key_is_order_independent():  # noqa: D401
    args1 = {"b": 2, "a": 1}
    args2 = {"a": 1, "b": 2}
    assert build_cache_key("my_tool", args1) == build_cache_key("my_tool", args2)


def test_extract_error_meta_raw_string():  # noqa: D401
    """When <meta> is not valid JSON it should be returned as raw text."""
    xml = (
        """
        <tool_response>
            <error code="X" retryable="false">
                <meta>raw-text</meta>
            </error>
        </tool_response>
        """
    )
    res = extract_error_from_xml(xml)
    assert res is not None and res["meta"] == "raw-text"


def test_extract_error_malformed_xml_returns_none():  # noqa: D401
    malformed = "<tool_response> <error></tool_response"  # missing closing bracket
    assert extract_error_from_xml(malformed) is None


def test_build_cache_key_with_tenant():  # noqa: D401
    key = build_cache_key("t", {"x": 1}, tenant_id="tenantA")
    assert key.startswith("tenant:tenantA:tool:t:")



@pytest.mark.asyncio
async def test_null_async_cm():  # noqa: D401
    """Ensure null_async_cm yields without blocking."""
    enter = False
    # Enter and exit without errors
    async with null_async_cm():
        enter = True
    assert enter is True


def test_extract_error_element_missing():  # noqa: D401
    xml = "<wrapper><errorx code='bad'></errorx></wrapper>"  # contains substring but no <error>
    assert extract_error_from_xml(xml) is None 