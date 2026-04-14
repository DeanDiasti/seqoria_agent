"""Tests for MLX-LM and MLX-VLM providers (mocked; no GPU)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from seqoria_agent.providers.errors import InvalidRequestError
from seqoria_agent.providers.schemas import LLMResponse


class TestMLXLMProviderMixin:
    def test_prepare_chat_messages_dict_user(self) -> None:
        from seqoria_agent.providers.mlx_lm_provider import _MLXLMProviderMixin

        m = _MLXLMProviderMixin()
        msgs = m._prepare_chat_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert msgs == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]


@pytest.mark.asyncio
async def test_mlx_lm_tools_rejected() -> None:
    from seqoria_agent.providers.mlx_lm_provider import AsyncMLXLMProvider

    with (
        patch("seqoria_agent.providers.mlx_lm_provider.mlx_lm_available", return_value=True),
        patch("seqoria_agent.providers.mlx_lm_provider._require_apple_silicon"),
        patch("seqoria_agent.providers.mlx_lm_provider._import_mlx_lm") as imp,
    ):
        imp.return_value = (MagicMock(), MagicMock(), MagicMock())
        p = AsyncMLXLMProvider("dummy-model", require_apple_silicon=False, lazy_load=True)

    with pytest.raises(InvalidRequestError, match="does not support tool calling"):
        await p.async_generate([{"role": "user", "content": "x"}], tools=[{"type": "function"}])


@pytest.mark.asyncio
async def test_async_mlx_lm_generate_mocked() -> None:
    from seqoria_agent.providers.mlx_lm_provider import AsyncMLXLMProvider

    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="FORMATTED")
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])

    model = MagicMock()

    def load_fn(path: str, **_: object) -> tuple[MagicMock, MagicMock]:
        return model, tokenizer

    def generate_fn(m: object, tok: object, *, prompt: str, **_: object) -> str:
        assert prompt == "FORMATTED"
        return "answer text"

    with (
        patch("seqoria_agent.providers.mlx_lm_provider.mlx_lm_available", return_value=True),
        patch("seqoria_agent.providers.mlx_lm_provider._require_apple_silicon"),
        patch(
            "seqoria_agent.providers.mlx_lm_provider._import_mlx_lm",
            return_value=(load_fn, generate_fn, MagicMock()),
        ),
    ):
        p = AsyncMLXLMProvider("hf/model", require_apple_silicon=False, lazy_load=True)
        resp = await p.async_generate([{"role": "user", "content": "Q"}])

    assert isinstance(resp, LLMResponse)
    assert resp.text == "answer text"
    assert resp.type == "llm_response"
    tokenizer.apply_chat_template.assert_called_once()


@pytest.mark.asyncio
async def test_async_mlx_lm_stream_mocked() -> None:
    from seqoria_agent.providers.mlx_lm_provider import AsyncMLXLMProvider

    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="PROMPT")
    tokenizer.encode = MagicMock(return_value=[1])

    class Chunk:
        def __init__(self, text: str) -> None:
            self.text = text

    def stream_gen(m: object, tok: object, prompt: str, **_: object):
        yield Chunk("a")
        yield Chunk("b")

    def load_fn(path: str, **_: object) -> tuple[MagicMock, MagicMock]:
        return MagicMock(), tokenizer

    with (
        patch("seqoria_agent.providers.mlx_lm_provider.mlx_lm_available", return_value=True),
        patch("seqoria_agent.providers.mlx_lm_provider._require_apple_silicon"),
        patch(
            "seqoria_agent.providers.mlx_lm_provider._import_mlx_lm",
            return_value=(load_fn, MagicMock(), stream_gen),
        ),
    ):
        p = AsyncMLXLMProvider("hf/model", require_apple_silicon=False, lazy_load=True)
        chunks = []
        async for c in p.async_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

    assert len(chunks) >= 2
    assert chunks[-1].type == "llm_response"
    assert chunks[-1].text == "ab"


@pytest.mark.asyncio
async def test_async_mlx_vlm_generate_mocked() -> None:
    from seqoria_agent.providers.mlx_vlm_provider import AsyncMLXVLMProvider

    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.encode = MagicMock(return_value=[1, 2])

    model = MagicMock()
    model.config = {"model_type": "test"}

    def load_fn(path: str) -> tuple[MagicMock, MagicMock]:
        return model, processor

    def apply_chat_template(
        proc: object,
        config: object,
        messages: object,
        *args: object,
        **kwargs: object,
    ) -> str:
        assert kwargs.get("num_images", 0) == 0
        assert kwargs.get("num_audios", 0) == 0
        return "VLM_PROMPT"

    class _GenResult:
        text = "vision answer"
        prompt_tokens = 1
        generation_tokens = 2
        total_tokens = 3

    def generate_fn(
        m: object,
        proc: object,
        prompt: str,
        image: object = None,
        audio: object = None,
        verbose: bool = False,
        **kwargs: object,
    ) -> object:
        assert prompt == "VLM_PROMPT"
        assert image is None
        assert audio is None
        return _GenResult()

    with (
        patch("seqoria_agent.providers.mlx_vlm_provider.mlx_vlm_available", return_value=True),
        patch("seqoria_agent.providers.mlx_vlm_provider._require_apple_silicon"),
        patch(
            "seqoria_agent.providers.mlx_vlm_provider._import_mlx_vlm",
            return_value=(load_fn, generate_fn, MagicMock(), apply_chat_template, None),
        ),
    ):
        p = AsyncMLXVLMProvider("hf/vlm", require_apple_silicon=False, lazy_load=True)
        resp = await p.async_generate([{"role": "user", "content": "Describe"}])

    assert resp.text == "vision answer"


def test_get_provider_mlx_lm() -> None:
    from seqoria_agent.providers import get_provider

    cls = get_provider("mlx_lm", async_variant=True)
    assert cls.__name__ == "AsyncMLXLMProvider"


def test_list_available_providers_has_mlx_keys() -> None:
    from seqoria_agent.providers import list_available_providers

    d = list_available_providers()
    assert "mlx_lm" in d
    assert "mlx_vlm" in d
    assert isinstance(d["mlx_lm"], bool)
    assert isinstance(d["mlx_vlm"], bool)


def test_mlx_vlm_multimodal_extracts_audio_and_images(tmp_path) -> None:
    from seqoria_agent.providers.mlx_vlm_provider import _MLXVLMProviderMixin

    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")

    m = _MLXVLMProviderMixin()
    content = [
        {"type": "text", "text": "What do you hear?"},
        {"type": "image", "url": str(png)},
        {"type": "audio", "url": str(wav)},
    ]

    text, images, audios = m._convert_multimodal_content_to_text_and_refs(content, str(tmp_path))
    assert "What do you hear?" in text
    assert str(png) in images
    assert str(wav) in audios
