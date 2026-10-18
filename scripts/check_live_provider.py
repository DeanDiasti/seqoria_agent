"""Check a real model/tool roundtrip in ordinary and streaming modes.

This opt-in check makes four model requests and may incur provider charges.
Credentials come from the environment; no prompts, responses, or secrets are logged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from contextlib import aclosing

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models import AssistantMessage, ChatChunk, ToolCallRequest, ToolCallResult
from seqoria_agent.providers import (
    AsyncAnthropicProvider,
    AsyncOllamaProvider,
    AsyncOpenAICompatibleProvider,
    AsyncOpenAIProvider,
)


def configured_provider(args):
    common = {"model_id": args.model, "max_retries": 0, "timeout": args.timeout}
    if args.provider == "openai":
        return AsyncOpenAIProvider(**common, max_output_tokens=256, store=False)
    if args.provider == "anthropic":
        return AsyncAnthropicProvider(**common, max_tokens=256)
    if args.provider == "compatible":
        if not os.environ.get("SEQORIA_BASE_URL") or not os.environ.get("SEQORIA_API_KEY"):
            raise ValueError("Set SEQORIA_BASE_URL and SEQORIA_API_KEY")
        return AsyncOpenAICompatibleProvider(
            **common,
            base_url=os.environ["SEQORIA_BASE_URL"],
            api_key=os.environ["SEQORIA_API_KEY"],
            max_tokens=256,
            reasoning_effort=args.reasoning_effort,
        )
    return AsyncOllamaProvider(**common, auto_pull=False, num_predict=256, think=False)


async def check(args):
    toolbox = LocalToolbox()
    invocations = []
    expected = "sum=42; proof=seqoria-launch-ok"

    @toolbox.tool(server_populated=["user_id"])
    async def launch_sum(a: int, b: int, user_id: str) -> str:
        """Add two integers and return the server's verification string."""
        assert (a, b, user_id) == (19, 23, "smoke-user")
        invocations.append(user_id)
        return expected

    prompt = (
        "Call launch_sum exactly once with a=19 and b=23. Do not compute the result yourself. "
        "After the tool returns, repeat its exact response, with no extra text."
    )
    modes = []
    async with (
        configured_provider(args) as provider,
        AsyncAIClient(providers=[provider], local_toolbox=toolbox) as client,
    ):
        for streaming in (False, True):
            history = []
            reply = None
            deltas = []
            method = client.stream_chat if streaming else client.chat
            async with aclosing(
                method(prompt, history=history, context={"user_id": "smoke-user"}, max_turns=2)
            ) as stream:
                async for event in stream:
                    if isinstance(event, AssistantMessage):
                        reply = event
                    elif isinstance(event, ChatChunk):
                        deltas.append(event.content or "")
            assert reply is not None and expected in str(reply.content), "Missing final tool proof"
            requests = [item for item in history if isinstance(item, ToolCallRequest)]
            results = [item for item in history if isinstance(item, ToolCallResult)]
            assert len(requests) == len(results) == 1, "Expected one paired tool call"
            assert requests[0].id == results[0].tool_call_id, "Tool result ID mismatch"
            assert results[0].result == expected, "Tool result changed"
            if streaming:
                assert expected in "".join(deltas), "Missing streamed final response"
            modes.append("streaming" if streaming else "ordinary")
    assert invocations == ["smoke-user", "smoke-user"], "Unexpected tool invocation count"
    return {"provider": args.provider, "model": args.model, "passed": modes}


def main():
    logging.basicConfig(level=logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=["openai", "anthropic", "compatible", "ollama"], required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--reasoning-effort", help="Optional compatible-endpoint reasoning setting")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    # Model and tool payloads are synthetic. Keep exception bodies out of shared logs.
    try:
        result = asyncio.run(asyncio.wait_for(check(args), timeout=args.timeout * 4))
    except Exception as exc:
        print(
            json.dumps(
                {"provider": args.provider, "status": "failed", "error_type": type(exc).__name__}
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
