"""Tests for vision/image support in providers."""

import pytest
import base64
from unittest.mock import patch, MagicMock


class TestOpenAIImageConversion:
    """Tests for OpenAI Responses API image content conversion."""

    def test_convert_text_content(self):
        """Text content should be converted to input_text format."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        content = [{"type": "text", "text": "Hello, world!"}]
        
        result = mixin._convert_image_content_responses(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Hello, world!"

    def test_convert_image_url(self):
        """Image with URL should be converted to Responses API input_image format."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        content = [{"type": "image", "url": "https://example.com/image.jpg"}]
        
        result = mixin._convert_image_content_responses(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://example.com/image.jpg"
        assert result[0]["detail"] == "auto"

    def test_convert_image_base64(self):
        """Image with base64 should be converted to data URL format."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        content = [{
            "type": "image",
            "base64": test_base64,
            "media_type": "image/png"
        }]
        
        result = mixin._convert_image_content_responses(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"].startswith("data:image/png;base64,")
        assert test_base64 in result[0]["image_url"]

    def test_convert_image_base64_default_media_type(self):
        """Image with base64 should default to image/jpeg media type."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        content = [{"type": "image", "base64": test_base64}]
        
        result = mixin._convert_image_content_responses(content)
        
        assert result[0]["image_url"].startswith("data:image/jpeg;base64,")

    def test_openai_legacy_format_converted(self):
        """Legacy OpenAI image_url format should be converted to Responses API format."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        content = [{
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }]
        
        result = mixin._convert_image_content_responses(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://example.com/image.jpg"

    def test_mixed_content(self):
        """Mixed text and image content should all be converted correctly."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        content = [
            {"type": "text", "text": "What's in these images?"},
            {"type": "image", "url": "https://example.com/image1.jpg"},
            {"type": "image", "base64": test_base64, "media_type": "image/png"}
        ]
        
        result = mixin._convert_image_content_responses(content)
        
        assert len(result) == 3
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "What's in these images?"
        assert result[1]["type"] == "input_image"
        assert result[1]["image_url"] == "https://example.com/image1.jpg"
        assert result[2]["type"] == "input_image"
        assert "data:image/png;base64," in result[2]["image_url"]

    def test_unknown_type_with_text(self):
        """Unknown content types with text field should be converted to input_text."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        content = [{"type": "unknown", "text": "some_data"}]
        
        result = mixin._convert_image_content_responses(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "some_data"


class TestAnthropicImageConversion:
    """Tests for Anthropic image content conversion."""

    def test_convert_text_content(self):
        """Text content should pass through unchanged."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        content = [{"type": "text", "text": "Hello, world!"}]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello, world!"

    def test_convert_image_url(self):
        """Image with URL should be converted to Anthropic format."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        content = [{"type": "image", "url": "https://example.com/image.jpg"}]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/image.jpg"

    def test_convert_image_base64(self):
        """Image with base64 should be converted to Anthropic format."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        content = [{
            "type": "image",
            "base64": test_base64,
            "media_type": "image/png"
        }]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert result[0]["source"]["data"] == test_base64

    def test_convert_image_base64_default_media_type(self):
        """Image with base64 should default to image/jpeg media type."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        content = [{"type": "image", "base64": test_base64}]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert result[0]["source"]["media_type"] == "image/jpeg"

    def test_anthropic_format_passthrough(self):
        """Content already in Anthropic format should pass through."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        content = [{
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.jpg"
            }
        }]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/image.jpg"

    def test_convert_openai_image_url_format(self):
        """OpenAI image_url format should be converted to Anthropic format."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        content = [{
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/image.jpg"

    def test_convert_openai_data_url_format(self):
        """OpenAI data URL format should be converted to Anthropic base64 format."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        content = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{test_base64}"}
        }]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert result[0]["source"]["data"] == test_base64

    def test_mixed_content(self):
        """Mixed text and image content should all be converted correctly."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        content = [
            {"type": "text", "text": "What's in these images?"},
            {"type": "image", "url": "https://example.com/image1.jpg"},
            {"type": "image", "base64": test_base64, "media_type": "image/png"}
        ]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "What's in these images?"
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "url"
        assert result[2]["type"] == "image"
        assert result[2]["source"]["type"] == "base64"

    def test_unknown_type_passthrough(self):
        """Unknown content types should pass through unchanged."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        content = [{"type": "unknown", "data": "some_data"}]
        
        result = mixin._convert_image_content_anthropic(content)
        
        assert len(result) == 1
        assert result[0] == {"type": "unknown", "data": "some_data"}


class TestOpenAIChatHistoryWithImages:
    """Tests for OpenAI Responses API chat history preparation with images."""

    def test_prepare_history_with_text_only(self):
        """Chat history with text-only content should work normally."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        instructions, input_items = mixin._prepare_input_responses(history)
        
        assert instructions is None
        assert len(input_items) == 2
        assert input_items[0]["type"] == "message"
        assert input_items[0]["role"] == "user"
        assert input_items[0]["content"] == "Hello"
        assert input_items[1]["type"] == "message"
        assert input_items[1]["role"] == "assistant"
        assert input_items[1]["content"] == "Hi there!"

    def test_prepare_history_with_image_url(self):
        """Chat history with image URL should be converted correctly."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/cat.jpg"}
            ]
        }]
        
        instructions, input_items = mixin._prepare_input_responses(history)
        
        assert len(input_items) == 1
        assert input_items[0]["type"] == "message"
        assert input_items[0]["role"] == "user"
        assert isinstance(input_items[0]["content"], list)
        assert len(input_items[0]["content"]) == 2
        assert input_items[0]["content"][0]["type"] == "input_text"
        assert input_items[0]["content"][1]["type"] == "input_image"
        assert input_items[0]["content"][1]["image_url"] == "https://example.com/cat.jpg"

    def test_prepare_history_with_image_base64(self):
        """Chat history with base64 image should be converted correctly."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image", "base64": test_base64, "media_type": "image/jpeg"}
            ]
        }]
        
        instructions, input_items = mixin._prepare_input_responses(history)
        
        assert len(input_items) == 1
        assert input_items[0]["content"][1]["type"] == "input_image"
        assert f"data:image/jpeg;base64,{test_base64}" == input_items[0]["content"][1]["image_url"]

    def test_prepare_history_tool_call_uses_call_id_without_invalid_id(self):
        """Tool-call history with internal call_* IDs should not emit invalid item IDs."""
        from seqoria_agent.models.chunk import ToolCallRequest
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

        mixin = _OpenAIProviderMixin()
        history = [
            ToolCallRequest(
                id="call_abc123",
                name="search_markets",
                arguments={"status": "open"},
            )
        ]

        _, input_items = mixin._prepare_input_responses(history)

        assert input_items[0]["type"] == "function_call"
        assert input_items[0]["call_id"] == "call_abc123"
        assert "id" not in input_items[0]

    def test_prepare_history_tool_call_keeps_fc_item_id_when_present(self):
        """Raw dict tool-call history keeps valid Responses item IDs."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

        mixin = _OpenAIProviderMixin()
        history = [
            {
                "role": "tool_call",
                "id": "fc_abc123",
                "call_id": "call_abc123",
                "name": "search_markets",
                "arguments": {"status": "open"},
            }
        ]

        _, input_items = mixin._prepare_input_responses(history)

        assert input_items[0]["type"] == "function_call"
        assert input_items[0]["id"] == "fc_abc123"
        assert input_items[0]["call_id"] == "call_abc123"


class TestAnthropicChatHistoryWithImages:
    """Tests for Anthropic chat history preparation with images."""

    def test_prepare_history_with_text_only(self):
        """Chat history with text-only content should work normally."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        system_msg, messages = mixin._prepare_chat_history_sync(history)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_prepare_history_with_image_url(self):
        """Chat history with image URL should be converted correctly."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/cat.jpg"}
            ]
        }]
        
        system_msg, messages = mixin._prepare_chat_history_sync(history)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image"
        assert messages[0]["content"][1]["source"]["type"] == "url"
        assert messages[0]["content"][1]["source"]["url"] == "https://example.com/cat.jpg"

    def test_prepare_history_with_image_base64(self):
        """Chat history with base64 image should be converted correctly."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        test_base64 = base64.b64encode(b"fake_image_data").decode()
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image", "base64": test_base64, "media_type": "image/jpeg"}
            ]
        }]
        
        system_msg, messages = mixin._prepare_chat_history_sync(history)
        
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/jpeg"
        assert content[1]["source"]["data"] == test_base64

    def test_prepare_history_multiple_images(self):
        """Chat history with multiple images should all be converted."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these images"},
                {"type": "image", "url": "https://example.com/image1.jpg"},
                {"type": "image", "url": "https://example.com/image2.jpg"},
                {"type": "image", "url": "https://example.com/image3.jpg"}
            ]
        }]
        
        system_msg, messages = mixin._prepare_chat_history_sync(history)
        
        content = messages[0]["content"]
        assert len(content) == 4
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "image"
        assert content[3]["type"] == "image"


class TestCrossProviderCompatibility:
    """Tests for cross-provider image format compatibility."""

    def test_openai_format_works_in_anthropic(self):
        """OpenAI image_url format should be usable with Anthropic provider."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        # OpenAI format
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ]
        }]
        
        system_msg, messages = mixin._prepare_chat_history_sync(history)
        
        # Should be converted to Anthropic format
        content = messages[0]["content"]
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "url"

    def test_common_format_works_in_both(self):
        """Common image format should work with both providers."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        # Common format
        common_content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image", "url": "https://example.com/photo.jpg"}
        ]
        
        # Test OpenAI Responses API format
        openai_mixin = _OpenAIProviderMixin()
        openai_result = openai_mixin._convert_image_content_responses(common_content.copy())
        assert openai_result[1]["type"] == "input_image"
        assert openai_result[1]["image_url"] == "https://example.com/photo.jpg"
        
        # Test Anthropic
        anthropic_mixin = _AnthropicProviderMixin()
        anthropic_result = anthropic_mixin._convert_image_content_anthropic(common_content.copy())
        assert anthropic_result[1]["type"] == "image"
        assert anthropic_result[1]["source"]["url"] == "https://example.com/photo.jpg"

