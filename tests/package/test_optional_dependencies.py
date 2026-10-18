"""Optional modules distinguish absent extras from broken dependencies."""

import pytest

from seqoria_agent.client.ai_client import utils


def test_optional_module_loading_only_suppresses_missing_requested_module(monkeypatch):

    def missing_optional(name):
        raise ModuleNotFoundError(name="optional")

    monkeypatch.setattr(utils, "import_module", missing_optional)
    assert utils._optional_module("optional") is None
    with pytest.raises(ModuleNotFoundError):
        utils._optional_module("installed_but_broken")
