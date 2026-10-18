"""Public exports reflect the supported feature set and removed modules stay absent."""

import importlib.util

import pytest

import seqoria_agent
from seqoria_agent import providers


def test_removed_local_adapters_are_absent_from_package():
    for name in ("mlx_lm", "mlx_vlm"):
        assert importlib.util.find_spec(f"seqoria_agent.providers.{name}_provider") is None
        with pytest.raises(ValueError, match="Unknown provider"):
            providers.get_provider(name)
    assert set(providers.list_available_providers()) == {
        "openai",
        "openai_compatible",
        "anthropic",
        "ollama",
    }
    assert not any(("MLX" in name for name in seqoria_agent.__all__ + providers.__all__))


def test_server_wrapper_is_not_exported():
    import seqoria_agent.models as models

    assert not hasattr(seqoria_agent, "SeqoriaMCP")
    assert "SeqoriaMCP" not in seqoria_agent.__all__
    assert not hasattr(models, "XMCPError")


def test_automatic_memory_is_removed_from_public_api():
    assert "AgenticMemory" not in seqoria_agent.__all__
    assert not hasattr(seqoria_agent, "AgenticMemory")
