"""examples/providers/ollama_stream.py.

Demonstration of using `OllamaProvider.stream` for streaming responses.

This example shows how to use the synchronous Ollama provider to stream
text completions from a locally running Ollama instance.

Prerequisites
-------------
1. Install Ollama from https://ollama.com/download
2. Start Ollama (it runs as a background service)
3. Pull a model: ``ollama pull qwen3-vl:4b``
4. Install this package: ``pip install seqoria-agent``

Run with::

    python examples/providers/ollama_stream.py

Vision Example
--------------
The script also demonstrates multimodal (vision) capabilities if you provide
an image path as an argument:

    python examples/providers/ollama_stream.py /path/to/image.jpg

Thinking Mode
-------------
Enable thinking/reasoning output with --think flag:

    python examples/providers/ollama_stream.py --think
"""

from __future__ import annotations

import base64
import sys

from seqoria_agent.providers import OllamaProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCallList


def main() -> None:
    """Run the Ollama streaming demo."""

    # Parse arguments
    args = sys.argv[1:]
    think_mode = "--think" in args
    if think_mode:
        args.remove("--think")

    # Initialize provider with qwen3-vl model (multimodal capable)
    # If the model isn't available, it will attempt to pull it automatically
    print("Initializing Ollama provider with qwen3-vl:4b...")
    print("(If model isn't downloaded, this will pull it automatically)\n")

    provider_kwargs = {
        "model_id": "qwen3-vl:4b",
        "validate_on_init": True,
        "temperature": 0.7,
    }

    # Enable thinking mode if requested
    if think_mode:
        provider_kwargs["think"] = True
        print("🧠 Thinking mode enabled - will show reasoning process\n")

    try:
        provider = OllamaProvider(**provider_kwargs)
    except Exception as e:
        print(f"Error initializing provider: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        sys.exit(1)

    with provider:
        # Check if user provided an image path for vision demo
        if args:
            image_path = args[0]
            run_vision_demo(provider, image_path, think_mode)
        else:
            run_text_demo(provider, think_mode)


def run_text_demo(provider: OllamaProvider, think_mode: bool = False) -> None:
    """Run a simple text streaming demo."""

    chat_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses concise.",
        },
        {
            "role": "user",
            "content": "Explain quantum computing in 3 sentences.",
        },
    ]

    print("=" * 60)
    print("Text Streaming Demo" + (" (with Thinking)" if think_mode else ""))
    print("=" * 60)
    print("\nUser: Explain quantum computing in 3 sentences.\n")

    in_thinking = False

    # Stream the response
    for chunk in provider.stream(chat_history=chat_history):
        if isinstance(chunk, LLMResponse):
            if chunk.type == "llm_delta":
                # Handle thinking output
                if chunk.thinking:
                    if not in_thinking:
                        print("🧠 Thinking: ", end="", flush=True)
                        in_thinking = True
                    print(chunk.thinking, end="", flush=True)

                # Handle regular text output
                if chunk.text:
                    if in_thinking:
                        print("\n\nAssistant: ", end="", flush=True)
                        in_thinking = False
                    elif not in_thinking and chunk.text == chunk.text.lstrip():
                        # First text chunk
                        if "Assistant:" not in str(chunk.text):
                            print("Assistant: ", end="", flush=True)
                    print(chunk.text, end="", flush=True)
            else:
                # Final response with usage info
                print("\n")
                print("-" * 40)
                if chunk.thinking:
                    print(f"Full thinking: {len(chunk.thinking)} chars")
                print(f"Stop reason: {chunk.stop_reason}")
                if chunk.usage:
                    print(
                        f"Tokens - Prompt: {chunk.usage.prompt_tokens}, "
                        f"Completion: {chunk.usage.completion_tokens}, "
                        f"Total: {chunk.usage.total_tokens}"
                    )
        elif isinstance(chunk, ToolCallList):
            print(f"\nTool calls: {[tc.function_name for tc in chunk.tool_calls]}")

    print()


def run_vision_demo(provider: OllamaProvider, image_path: str, think_mode: bool = False) -> None:
    """Run a vision/multimodal streaming demo."""

    print("=" * 60)
    print("Vision Streaming Demo" + (" (with Thinking)" if think_mode else ""))
    print("=" * 60)
    print(f"\nLoading image: {image_path}\n")

    # Load and encode the image
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Determine media type from extension
    ext = image_path.lower().split(".")[-1]
    media_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    chat_history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image? Describe it in detail."},
                {"type": "image", "base64": image_base64, "media_type": media_type},
            ],
        },
    ]

    print("User: What do you see in this image? Describe it in detail.\n")

    in_thinking = False

    # Stream the response
    for chunk in provider.stream(chat_history=chat_history):
        if isinstance(chunk, LLMResponse):
            if chunk.type == "llm_delta":
                # Handle thinking output
                if chunk.thinking:
                    if not in_thinking:
                        print("🧠 Thinking: ", end="", flush=True)
                        in_thinking = True
                    print(chunk.thinking, end="", flush=True)

                # Handle regular text output
                if chunk.text:
                    if in_thinking:
                        print("\n\nAssistant: ", end="", flush=True)
                        in_thinking = False
                    elif not in_thinking and chunk.text == chunk.text.lstrip():
                        if "Assistant:" not in str(chunk.text):
                            print("Assistant: ", end="", flush=True)
                    print(chunk.text, end="", flush=True)
            else:
                print("\n")
                print("-" * 40)
                if chunk.thinking:
                    print(f"Full thinking: {len(chunk.thinking)} chars")
                print(f"Stop reason: {chunk.stop_reason}")
                if chunk.usage:
                    print(
                        f"Tokens - Prompt: {chunk.usage.prompt_tokens}, "
                        f"Completion: {chunk.usage.completion_tokens}, "
                        f"Total: {chunk.usage.total_tokens}"
                    )

    print()


if __name__ == "__main__":
    main()
