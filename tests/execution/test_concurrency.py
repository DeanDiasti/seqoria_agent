"""Shared execution limits bound work across concurrent requests."""

import asyncio

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models.chunk import ToolCallRequest
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("inf"), float("nan")])
def test_concurrency_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_concurrency"):
        AsyncAIClient(providers=[ReplyProvider()], max_concurrency=value)


def request(name, **arguments):
    return ToolCallRequest(id=name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_concurrency_limit_is_shared_between_requests_and_released_on_cancellation():
    box = LocalToolbox()
    started = asyncio.Queue()
    active = peak = 0
    release = asyncio.Event()

    @box.tool()
    async def wait(label: str):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.put_nowait(label)
        try:
            await release.wait()
            return label
        finally:
            active -= 1

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, max_concurrency=1
    ) as client:
        first = asyncio.create_task(client.run_tool_calls([request("wait", label="first")]))
        assert await asyncio.wait_for(started.get(), 1) == "first"
        second = asyncio.create_task(client.run_tool_calls([request("wait", label="second")]))
        await asyncio.sleep(0)
        assert started.empty()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.wait_for(started.get(), 1) == "second"
        release.set()
        assert (await asyncio.wait_for(second, 1))[0].result == "second"
    assert active == 0
    assert peak == 1


@pytest.mark.asyncio
async def test_parameter_resolution_obeys_concurrency_limit_across_requests():
    box = LocalToolbox()
    started = asyncio.Queue()
    release = asyncio.Event()
    active = peak = 0

    async def resolve_user_id(label):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.put_nowait(label)
        try:
            await release.wait()
            return label
        finally:
            active -= 1

    @box.tool(server_populated=["user_id"])
    async def protected(user_id: str):
        return user_id

    async with AsyncAIClient(
        providers=[ReplyProvider()],
        local_toolbox=box,
        max_concurrency=2,
        param_registry={"user_id": resolve_user_id},
    ) as client:
        tasks = [
            asyncio.create_task(
                client.run_tool_calls([request("protected")], context={"label": str(i)})
            )
            for i in range(3)
        ]
        try:
            labels = {await asyncio.wait_for(started.get(), 1) for _ in range(2)}
            assert len(labels) == 2
            await asyncio.sleep(0)
            assert started.empty()
            assert active == peak == 2
        finally:
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert [result[0].result for result in results] == ["0", "1", "2"]
    assert active == 0
    assert peak == 2


@pytest.mark.asyncio
async def test_sync_tool_does_not_block_other_requests():
    import threading

    box = LocalToolbox()
    release = threading.Event()
    started = threading.Event()

    @box.tool()
    def blocking():
        started.set()
        return release.wait(timeout=2)

    task = asyncio.create_task(box.call_tool("blocking", {}))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert not task.done()
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
    finally:
        release.set()
        await task
