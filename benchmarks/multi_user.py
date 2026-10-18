"""Measure the local multi-user agent loop with deterministic HTTP responses.

No requests leave the process. This measures orchestration overhead and tool
queueing, not the latency or capacity of a hosted model or database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import time
from importlib.metadata import version
from typing import Any

import httpx

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models import AssistantMessage, ToolCallRequest
from seqoria_agent.providers import AsyncOpenAICompatibleProvider


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    tools = LocalToolbox()
    active = 0
    peak = 0
    executions = 0
    model_calls = 0
    probe_started = asyncio.Event()
    probe_cancelled = asyncio.Event()
    existing_tasks = asyncio.all_tasks()

    @tools.tool(server_populated=["user_id"])
    async def lookup(user_id: str) -> dict[str, str]:
        nonlocal active, peak, executions
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(args.tool_latency_ms / 1000)
            executions += 1
            return {"user_id": user_id}
        finally:
            active -= 1

    @tools.tool(server_populated=["user_id"])
    async def cancel_probe(user_id: str) -> str:
        nonlocal active
        active += 1
        probe_started.set()
        try:
            await asyncio.Event().wait()
            return user_id
        finally:
            active -= 1
            probe_cancelled.set()

    async def model(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        model_calls += 1
        call_id = f"lookup-{model_calls}"
        await asyncio.sleep(0)
        messages = json.loads(request.content)["messages"]
        if messages[-1]["role"] == "tool":
            message = {"role": "assistant", "content": messages[-1]["content"]}
            reason = "stop"
        else:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
            reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "model": "synthetic",
                "choices": [{"message": message, "finish_reason": reason}],
            },
        )

    durations: list[float] = []
    # Inject an in-process HTTP transport; close the provider's original pool.
    async with AsyncOpenAICompatibleProvider(
        model_id="synthetic", api_key="benchmark", max_retries=0
    ) as provider:
        await provider.async_client.aclose()
        provider.async_client = httpx.AsyncClient(
            base_url="https://benchmark.invalid", transport=httpx.MockTransport(model)
        )
        async with AsyncAIClient(
            providers=[provider],
            local_toolbox=tools,
            max_concurrency=args.max_concurrency,
            tool_timeout=30,
        ) as client:

            async def conversation(number: int) -> None:
                user_id = f"user-{number}"
                history: list[Any] = []
                for turn in range(args.rounds):
                    start = time.perf_counter()
                    events = [
                        event
                        async for event in client.chat(
                            f"lookup {turn}", history=history, context={"user_id": user_id}
                        )
                    ]
                    durations.append((time.perf_counter() - start) * 1000)
                    reply = events[-1]
                    assert isinstance(reply, AssistantMessage), "Missing final reply"
                    assert json.loads(reply.content) == {"user_id": user_id}, "Identity leaked"

            started = time.perf_counter()
            await asyncio.gather(*(conversation(number) for number in range(args.users)))
            elapsed = time.perf_counter() - started
            expected = args.users * args.rounds
            assert executions == expected
            assert model_calls == expected * 2
            assert 0 < peak <= args.max_concurrency
            assert active == 0

            # Cancellation must finish tool cleanup and release its execution slot.
            task = asyncio.create_task(
                client.run_tool_calls(
                    [ToolCallRequest(id="cancel", name="cancel_probe")],
                    context={"user_id": "cancel-user"},
                )
            )
            await asyncio.wait_for(probe_started.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert task.cancelled() and probe_cancelled.is_set() and active == 0
            recovered = await client.run_tool_calls(
                [ToolCallRequest(id="recovery", name="lookup")],
                context={"user_id": "recovery-user"},
            )
            assert json.loads(recovered[0].result) == {"user_id": "recovery-user"}
        assert not client.pool.connections
    assert provider.async_client.is_closed
    remaining = [task for task in asyncio.all_tasks() - existing_tasks if not task.done()]
    assert not remaining, "Client left background tasks running"
    return {
        "scope": "synthetic in-process HTTP model and async local tools; no hosted services",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seqoria_agent": version("seqoria-agent"),
        },
        "configuration": vars(args),
        "measurements": {
            "requests": expected,
            "elapsed_seconds": round(elapsed, 4),
            "requests_per_second": round(expected / elapsed, 2),
            "latency_ms": {
                "mean": round(statistics.mean(durations), 3),
                "p50": round(percentile(durations, 0.50), 3),
                "p95": round(percentile(durations, 0.95), 3),
                "p99": round(percentile(durations, 0.99), 3),
            },
            "peak_active_tools": peak,
        },
        "checks": {
            "identity_isolation": True,
            "concurrency_limit": True,
            "cancellation_cleanup": True,
            "post_cancellation_recovery": True,
            "http_client_closed": True,
            "no_background_tasks": True,
        },
    }


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative and finite")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=positive_int, default=100)
    parser.add_argument("--rounds", type=positive_int, default=5)
    parser.add_argument("--tool-latency-ms", type=nonnegative_float, default=5)
    parser.add_argument("--max-concurrency", type=positive_int, default=16)
    print(json.dumps(asyncio.run(benchmark(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
