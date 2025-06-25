from seqoria_agent.server.xmcp_core.formatter import get_formatter, register_formatter
from seqoria_agent.server.formatters import DeclarativeXMLFormatter


def test_get_formatter_dict_returns_declarative():
    sample = {"answer": "42"}
    fmt = get_formatter(sample)
    assert isinstance(fmt, DeclarativeXMLFormatter)


def test_custom_formatter_registration_and_caching():
    class Dummy:
        pass

    class DummyFmt(DeclarativeXMLFormatter):
        pass

    dummy_fmt_instance = DummyFmt()

    # Register custom formatter
    register_formatter(Dummy, dummy_fmt_instance)

    # First call returns the registered instance
    fmt1 = get_formatter(Dummy())
    assert fmt1 is dummy_fmt_instance

    # Subsequent call uses cache (same object)
    fmt2 = get_formatter(Dummy())
    assert fmt1 is fmt2 