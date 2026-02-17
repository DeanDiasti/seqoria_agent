from __future__ import annotations

import sys
import types

import pytest

from seqoria_agent.memory.embeddings.ollama import OllamaEmbedder


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch, async_client: type) -> None:
    module = types.ModuleType("ollama")
    module.AsyncClient = async_client
    monkeypatch.setitem(sys.modules, "ollama", module)


@pytest.mark.asyncio
async def test_embed_text_uses_embed_endpoint(monkeypatch: pytest.MonkeyPatch):
    class FakeAsyncClient:
        last_instance = None

        def __init__(self, host=None, headers=None):
            self.host = host
            self.headers = headers
            self.calls: list[tuple[str, list[str], dict]] = []
            FakeAsyncClient.last_instance = self

        async def embed(self, *, model, input, **kwargs):
            self.calls.append((model, input, kwargs))
            return {"embeddings": [[0.1, 0.2]]}

    _install_fake_ollama(monkeypatch, FakeAsyncClient)
    monkeypatch.setenv("OLLAMA_HOST", "http://fake-host:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")

    embedder = OllamaEmbedder(model="nomic-embed-text", truncate=True)
    vector = await embedder.embed_text("hello")

    assert vector == [0.1, 0.2]
    assert FakeAsyncClient.last_instance is not None
    assert FakeAsyncClient.last_instance.host == "http://fake-host:11434"
    assert FakeAsyncClient.last_instance.headers == {"Authorization": "Bearer env-key"}
    assert FakeAsyncClient.last_instance.calls == [
        ("nomic-embed-text", ["hello"], {"truncate": True})
    ]


@pytest.mark.asyncio
async def test_embed_texts_uses_legacy_embeddings_fallback(monkeypatch: pytest.MonkeyPatch):
    class FakeAsyncClient:
        def __init__(self, host=None, headers=None):
            self.calls: list[tuple[str, str]] = []

        async def embeddings(self, *, model, prompt):
            self.calls.append((model, prompt))
            return {"embedding": [float(len(prompt))]}

    _install_fake_ollama(monkeypatch, FakeAsyncClient)

    embedder = OllamaEmbedder(model="legacy-model")
    vectors = await embedder.embed_texts(["a", "abcd"])

    assert vectors == [[1.0], [4.0]]


@pytest.mark.asyncio
async def test_embed_texts_raises_when_dependency_missing(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("ollama")
    monkeypatch.setitem(sys.modules, "ollama", module)

    embedder = OllamaEmbedder()
    with pytest.raises(RuntimeError) as exc_info:
        await embedder.embed_texts(["hello"])

    assert "requires optional dependency 'ollama'" in str(exc_info.value)
