"""Verify a wheel installation from outside the source checkout."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from pathlib import Path

import seqoria_agent
from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.providers import (
    AsyncAnthropicProvider,
    AsyncOllamaProvider,
    AsyncOpenAICompatibleProvider,
    AsyncOpenAIProvider,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    installed = Path(seqoria_agent.__file__).resolve()
    assert not installed.is_relative_to(root), f"Imported checkout instead of wheel: {installed}"
    assert installed.with_name("py.typed").is_file()
    for name in (
        "seqoria_agent.memory",
        "seqoria_agent.server",
        "seqoria_agent.client.cache",
        "seqoria_agent.providers.mlx_lm_provider",
        "seqoria_agent.providers.mlx_vlm_provider",
    ):
        assert importlib.util.find_spec(name) is None, name
    for exported in (
        AsyncAIClient,
        LocalToolbox,
        AsyncOpenAIProvider,
        AsyncOpenAICompatibleProvider,
        AsyncAnthropicProvider,
        AsyncOllamaProvider,
    ):
        assert exported is not None
    print(
        f"Verified installed seqoria-agent {version('seqoria-agent')} core, HTTP providers, and typed marker."
    )


if __name__ == "__main__":
    main()
