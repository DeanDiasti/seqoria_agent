from __future__ import annotations

import pytest

from seqoria_agent.memory.embeddings.ollama import OllamaEmbedder


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_embed_text_uses_embed_endpoint(monkeypatch: pytest.MonkeyPatch):
    class FakeAsyncClient:
        last_instance = None

        def __init__(self, *, base_url=None, headers=None, timeout=None, http2=None):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout
            self.http2 = http2
            self.calls: list[tuple[str, dict]] = []
            FakeAsyncClient.last_instance = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, path, *, json):
            self.calls.append((path, json))
            return _FakeResponse({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setenv("OLLAMA_HOST", "http://fake-host:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    monkeypatch.setattr("seqoria_agent.memory.embeddings.ollama.httpx.AsyncClient", FakeAsyncClient)

    embedder = OllamaEmbedder(model="nomic-embed-text", truncate=True)
    vector = await embedder.embed_text("hello")

    assert vector == [0.1, 0.2]
    assert FakeAsyncClient.last_instance is not None
    assert FakeAsyncClient.last_instance.base_url == "http://fake-host:11434"
    assert FakeAsyncClient.last_instance.headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer env-key",
    }
    assert FakeAsyncClient.last_instance.calls == [
        (
            "/api/embed",
            {"model": "nomic-embed-text", "input": ["hello"], "truncate": True},
        )
    ]


@pytest.mark.asyncio
async def test_embed_texts_raises_on_unexpected_vector_count(monkeypatch: pytest.MonkeyPatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, path, *, json):
            return _FakeResponse({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr("seqoria_agent.memory.embeddings.ollama.httpx.AsyncClient", FakeAsyncClient)

    embedder = OllamaEmbedder(model="embed-model")
    with pytest.raises(RuntimeError) as exc_info:
        await embedder.embed_texts(["a", "b"])

    assert "unexpected number of vectors" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_texts_raises_on_invalid_payload(monkeypatch: pytest.MonkeyPatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, path, *, json):
            return _FakeResponse({"embedding": [0.1, 0.2]})

    monkeypatch.setattr("seqoria_agent.memory.embeddings.ollama.httpx.AsyncClient", FakeAsyncClient)

    embedder = OllamaEmbedder(model="embed-model")
    with pytest.raises(RuntimeError) as exc_info:
        await embedder.embed_texts(["hello"])

    assert "did not contain an 'embeddings' list" in str(exc_info.value)

