from typing import List

import pytest

from seqoria_agent.client import AIClient, ChatChunk  # type: ignore
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class DummyProvider(BaseProvider):
    """Provider that emits constant text *without* any tool calls."""

    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        return LLMResponse(model_id="dummy", text="steady", tool_calls=[])

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        async def _gen():
            for ch in "steady":
                yield ch
            yield LLMResponse(model_id="dummy", text="steady", tool_calls=[])
        return _gen()


# ---------------------------------------------------------------------------
# Helper – snapshot of the *previous* AIClient.chat implementation
# ---------------------------------------------------------------------------

async def legacy_chat(client: AIClient, user_query: str, history: List[dict], *, max_turns: int = 5):  # noqa: D401
    if history is None:
        raise ValueError("history required")
    if client._hist is not None:  # pylint: disable=protected-access
        raise RuntimeError("chat re-entered")
    client._hist = history  # pylint: disable=protected-access
    try:
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": client._system_prompt})  # pylint: disable=protected-access
        history.append({"role": "user", "content": user_query})
        if not client.tools_enabled():
            r = await client.llm.generate(history=history)  # type: ignore[arg-type]
            history.append({"role": "assistant", "content": r.text})
            return {"answer": r.text, "display": None, "history": history}
        for _ in range(max_turns):
            r = await client.llm.generate(history=history, tools=client.registry.llm_tools, tool_choice="auto")  # type: ignore[arg-type]
            history.append({
                "role": "assistant",
                "content": r.text,
                **({"tool_calls": [tc.model_dump() for tc in r.tool_calls]} if r.tool_calls else {}),
            })
            if not r.tool_calls:
                return {"answer": r.text, "display": None, "history": history}
            early = await client.executor.run_calls(tool_calls=r.tool_calls, history=history)
            if early:
                early["history"] = history
                return early
        return {"answer": "[Turn limit reached]", "display": None, "history": history}
    finally:
        client._hist = None  # pylint: disable=protected-access


async def legacy_stream_chat(
    client: AIClient,
    user_query: str,
    history: List[dict],
    *,
    max_turns: int = 5,
):  # noqa: D401
    if history is None:
        raise ValueError("history required")
    if client._hist is not None:  # pylint: disable=protected-access
        raise RuntimeError("stream_chat re-entered")
    client._hist = history  # pylint: disable=protected-access
    try:
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": client._system_prompt})  # pylint: disable=protected-access
        history.append({"role": "user", "content": user_query})

        if not client.tools_enabled():
            gen = await client._providers[0].async_stream(chat_history=history)  # pylint: disable=protected-access
            async for d in gen:
                if isinstance(d, str):
                    yield ChatChunk(type="llm_delta", text=d)
            return

        for _ in range(max_turns):
            agg: List[str] = []
            final: LLMResponse | None = None
            async for c in client.llm.stream(history=history, tools=client.registry.llm_tools, tool_choice="auto"):
                if isinstance(c, str):
                    agg.append(c)
                    yield ChatChunk(type="llm_delta", text=c)
                else:
                    final = c
            if final is None:
                final = LLMResponse(model_id="unknown", text="".join(agg))
            history.append({
                "role": "assistant",
                "content": final.text,
                **({"tool_calls": [tc.model_dump() for tc in final.tool_calls]} if final.tool_calls else {}),
            })
            if not final.tool_calls:
                return
            early = await client.executor.run_calls(tool_calls=final.tool_calls, history=history)
            if early:
                if early.get("display"):
                    yield ChatChunk(type="tool_display", envelope=early["display"])
                if early.get("answer"):
                    yield ChatChunk(type="llm_delta", text=early["answer"])
                return
        yield ChatChunk(type="llm_delta", text="[Turn limit reached]")
    finally:
        client._hist = None  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_parity():  # noqa: D401
    ai = AIClient(urls=[], providers=[DummyProvider()])
    async with ai as client:
        hist1: list[dict] = []
        hist2: list[dict] = []

        old_res = await legacy_chat(client, "hi", hist1)
        new_res = await client.chat("hi", hist2)

        assert old_res == new_res


@pytest.mark.asyncio
async def test_stream_chat_parity():  # noqa: D401
    ai = AIClient(urls=[], providers=[DummyProvider()])
    async with ai as client:
        hist_a: list[dict] = []
        hist_b: list[dict] = []

        async def collect(gen):
            out: List[ChatChunk] = []
            async for chunk in gen:
                out.append(chunk)
            return out

        old_chunks = await collect(legacy_stream_chat(client, "go", hist_a))
        new_chunks = await collect(client.stream_chat("go", hist_b))

        assert old_chunks == new_chunks 