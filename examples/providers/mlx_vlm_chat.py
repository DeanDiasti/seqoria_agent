"""examples/providers/mlx_vlm_chat.py

Terminal chat with a local vision-language model via :class:`MLXVLMProvider`
(MLX-VLM on Apple Silicon).

Uses ``mlx-community/gemma-3n-E2B-it-4bit`` by default (Gemma 3n E2B multimodal).

Prerequisites
-------------
- Apple Silicon Mac (arm64)
- ``pip install 'seqoria-agent[mlx-vlm]'`` (brings in ``mlx-vlm`` and dependencies)
- Enough disk/RAM for the 4-bit checkpoint; first run downloads weights from Hugging Face

Responses stream token-by-token via :meth:`MLXVLMProvider.stream` (backed by
``mlx_vlm.generate.stream_generate``).

Run::

    python examples/providers/mlx_vlm_chat.py

Optional model override::

    python examples/providers/mlx_vlm_chat.py --model mlx-community/another-vlm-4bit

In the REPL
-----------
- Type messages and press Enter.
- ``/image <path-or-URL>`` — attach that image to your **next** message.
- ``/audio <path-or-URL>`` — attach audio (e.g. ``.wav``, ``.mp3``) to your **next** message (local file or ``https://``).
- You can queue **both** ``/image`` and ``/audio`` before sending; they are combined on the next line.
- ``/quit`` or Ctrl+D — exit.

The provider passes ``image`` and ``audio`` into ``stream_generate`` (and sets
``num_images`` / ``num_audios`` on ``apply_chat_template``), matching the upstream MLX-VLM API.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

from seqoria_agent.providers import MLXVLMProvider
from seqoria_agent.providers.mlx_vlm_provider import mlx_vlm_available
from seqoria_agent.providers.schemas import LLMResponse

DEFAULT_MODEL = "mlx-community/gemma-3n-E2B-it-4bit"


def _append_image_block(blocks: List[dict[str, Any]], ref: str) -> None:
    ref = ref.strip()
    if ref.startswith(("http://", "https://")):
        blocks.append({"type": "image_url", "image_url": {"url": ref}})
    else:
        blocks.append({"type": "image", "url": ref})


def _append_audio_block(blocks: List[dict[str, Any]], ref: str) -> None:
    ref = ref.strip()
    if ref.startswith(("http://", "https://")):
        blocks.append({"type": "audio_url", "audio_url": {"url": ref}})
    else:
        blocks.append({"type": "audio", "url": ref})


def _build_user_content(
    text: str,
    image_ref: str | None,
    audio_ref: str | None,
) -> str | List[dict[str, Any]]:
    if not image_ref and not audio_ref:
        return text
    blocks: List[dict[str, Any]] = [{"type": "text", "text": text}]
    if image_ref:
        _append_image_block(blocks, image_ref)
    if audio_ref:
        _append_audio_block(blocks, audio_ref)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal VLM chat using MLX-VLM (local Apple Silicon inference).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face repo id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--no-apple-check",
        action="store_true",
        help="Skip macOS arm64 check (for debugging only; MLX needs Apple Silicon).",
    )
    args = parser.parse_args()

    if not mlx_vlm_available():
        print(
            "mlx_vlm is not installed. Install with:\n"
            "  pip install 'seqoria-agent[mlx-vlm]'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Model: {args.model}")
    print("Initializing provider (first inference may download weights and compile)…\n")

    try:
        provider = MLXVLMProvider(
            model_id=args.model,
            lazy_load=True,
            require_apple_silicon=not args.no_apple_check,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(f"Failed to create provider: {exc}", file=sys.stderr)
        sys.exit(1)

    history: List[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and concisely.",
        }
    ]

    print("Commands:")
    print("  /image <path-or-URL>  — attach image to your next message")
    print("  /audio <path-or-URL>  — attach audio to your next message")
    print("  /quit                 — exit\n")

    pending_image: str | None = None
    pending_audio: str | None = None

    while True:
        try:
            user_line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_line:
            continue

        lowered = user_line.lower()
        if lowered in ("/quit", "/exit", "quit", "exit"):
            break

        if lowered.startswith("/image "):
            rest = user_line[7:].strip().strip('"').strip("'")
            if rest.startswith(("http://", "https://")):
                pending_image = rest
                print("(Next message will include image URL.)\n")
            elif Path(rest).expanduser().is_file():
                pending_image = str(Path(rest).expanduser().resolve())
                print(f"(Next message will include image: {pending_image})\n")
            else:
                print(f"Not a file or URL: {rest}\n", file=sys.stderr)
            continue

        if lowered.startswith("/audio "):
            rest = user_line[7:].strip().strip('"').strip("'")
            if rest.startswith(("http://", "https://")):
                pending_audio = rest
                print("(Next message will include audio URL.)\n")
            elif Path(rest).expanduser().is_file():
                pending_audio = str(Path(rest).expanduser().resolve())
                print(f"(Next message will include audio: {pending_audio})\n")
            else:
                print(f"Not a file or URL: {rest}\n", file=sys.stderr)
            continue

        user_content = _build_user_content(user_line, pending_image, pending_audio)
        pending_image = None
        pending_audio = None
        history.append({"role": "user", "content": user_content})

        try:
            print("Assistant: ", end="", flush=True)
            full_text = ""
            for chunk in provider.stream(chat_history=history):
                if not isinstance(chunk, LLMResponse):
                    continue
                if chunk.type == "llm_delta" and chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_text += chunk.text
                elif chunk.type == "llm_response":
                    full_text = (chunk.text or full_text or "").strip()
                    if chunk.usage:
                        u = chunk.usage
                        print(
                            f"\n  (tokens: prompt={u.prompt_tokens}, "
                            f"completion={u.completion_tokens}, total={u.total_tokens})"
                        )
            print()
            history.append({"role": "assistant", "content": full_text})
        except Exception as exc:
            print(f"\n[error] {exc}\n", file=sys.stderr)
            history.pop()
            continue


if __name__ == "__main__":
    main()
