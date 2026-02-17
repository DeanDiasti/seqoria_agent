from __future__ import annotations

"""Comprehensive AgenticMemory demo with graph + retrieval insights.

This script demonstrates:
- Memory-first orchestration with llm_stream_once()
- Thinking-token streaming from Ollama (`think=True`)
- Tool-call cycle recording into memory
- Long-term memory writes and graph edge growth
- Checkpoints and retrieval
- Persistence round-trip

Run:
    uv run examples/manual_loop/agentic_memory_insights.py

Required Ollama models:
    ollama pull qwen3-vl:4b
    ollama pull embeddinggemma:latest
"""

import asyncio
import os
import sys
from pathlib import Path

from seqoria_agent import AsyncOllamaProvider
from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import (
    AgenticMemory,
    AgenticMemoryConfig,
    ContextConfig,
    MemoryConfig,
    PlanConfig,
    ToolHandlerConfig,
)
from seqoria_agent.memory.embeddings.ollama import OllamaEmbedder
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk, ToolCallRequestList
from seqoria_agent.toolbox import LocalToolbox


toolbox = LocalToolbox()

ENABLE_COLOR = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM", "").lower() != "dumb"
)
THINKING_COLOR_START = "\033[90m" if ENABLE_COLOR else ""
TOOL_CALL_COLOR_START = "\033[32m" if ENABLE_COLOR else ""
COLOR_RESET = "\033[0m" if ENABLE_COLOR else ""


@toolbox.tool(description="Fetch sample sales snapshot by region and quarter.")
def sales_snapshot(region: str, quarter: str) -> dict[str, float]:
    """Return deterministic sample metrics for demo repeatability."""
    table: dict[tuple[str, str], dict[str, float]] = {
        ("EMEA", "Q3"): {"revenue_m": 128.4, "cost_m": 81.1, "growth_pct": 8.9},
        ("EMEA", "Q4"): {"revenue_m": 133.7, "cost_m": 85.0, "growth_pct": 4.1},
        ("NA", "Q3"): {"revenue_m": 161.2, "cost_m": 101.3, "growth_pct": 6.7},
    }
    return table.get((region.upper(), quarter.upper()), {"revenue_m": 0.0, "cost_m": 0.0, "growth_pct": 0.0})


@toolbox.tool(description="Compute margin percentage from revenue and cost values.")
def margin_pct(revenue_m: float, cost_m: float) -> float:
    if revenue_m <= 0:
        return 0.0
    return round(((revenue_m - cost_m) / revenue_m) * 100.0, 2)


def print_memory_insights(memory: AgenticMemory, turn_name: str) -> None:
    """Print concise state-level diagnostics after each turn."""
    snapshot = memory.to_dict()
    raw_messages = snapshot["context"]["raw_messages"]
    nodes = snapshot["graph"]["nodes"]
    edges = snapshot["graph"]["edges"]
    checkpoints = snapshot["checkpoints"]
    compressed = snapshot["context"]["compressed_history"]

    print(f"\n[Memory Insight] {turn_name}")
    print(f"- Timeline events: {len(raw_messages)}")
    print(f"- Long-term nodes: {len(nodes)}")
    print(f"- Graph edges: {len(edges)}")
    print(f"- Checkpoints: {len(checkpoints)}")
    print(f"- Has compressed history: {bool(compressed)}")

    if nodes:
        latest = nodes[-1]
        print(
            "- Latest node:",
            f"type={latest.get('memory_type')}",
            f"id={latest.get('id')}",
            f"text={str(latest.get('compressed', ''))[:120]}",
        )
    if edges:
        last_edge = edges[-1]
        print(
            "- Latest edge:",
            f"{last_edge.get('source_id')} -> {last_edge.get('target_id')}",
            f"rel={last_edge.get('relationship')}",
            f"weight={last_edge.get('weight')}",
        )


async def run_stream_turn(
    ai: AsyncAIClient,
    memory: AgenticMemory,
    user_query: str,
) -> str:
    """Single memory-first streamed turn with tool loop support."""
    llm_history = memory.begin_turn(user_query, retrieval_query=user_query)
    final_text = ""

    for _ in range(6):
        streamed_text_parts: list[str] = []
        tool_turn = False
        thinking_stream_open = False
        assistant_stream_open = False
        print(f"\nYou: {user_query}", flush=True)

        async for chunk in ai.llm_stream_once(history=llm_history, use_tools=True):
            if isinstance(chunk, ChatChunk):
                if chunk.thinking:
                    if not thinking_stream_open:
                        print(f"{THINKING_COLOR_START}[thinking] ", end="", flush=True)
                        thinking_stream_open = True
                    print(chunk.thinking, end="", flush=True)
                if chunk.content:
                    if not assistant_stream_open:
                        if thinking_stream_open:
                            if COLOR_RESET:
                                print(COLOR_RESET, end="", flush=True)
                            print()
                        print("Assistant: ", end="", flush=True)
                        assistant_stream_open = True
                    print(chunk.content, end="", flush=True)
                    streamed_text_parts.append(chunk.content)
                continue

            if isinstance(chunk, ToolCallRequestList):
                if thinking_stream_open or assistant_stream_open:
                    if thinking_stream_open and COLOR_RESET:
                        print(COLOR_RESET, end="", flush=True)
                    print()
                print(f"\n\n{TOOL_CALL_COLOR_START}[tool-calls]{COLOR_RESET}")
                for call in chunk.tool_calls:
                    print(
                        f"{TOOL_CALL_COLOR_START}- {call.name} "
                        f"id={call.id} args={call.arguments}{COLOR_RESET}"
                    )

                tool_results = await ai.run_tool_calls(tool_calls=chunk.tool_calls)
                print("[tool-results]")
                for result in tool_results:
                    print(
                        f"- {result.name} id={result.tool_call_id} "
                        f"type={result.role} result={str(result.result)[:120]}"
                    )

                memory.record_tool_cycle(chunk.tool_calls, tool_results)
                llm_history = memory.messages_for_llm(query=user_query)
                tool_turn = True
                break

            if isinstance(chunk, AssistantMessage):
                final_text = chunk.content or "".join(streamed_text_parts)
                if chunk.thinking and not thinking_stream_open:
                    print(
                        f"\n{THINKING_COLOR_START}[final-thinking] "
                        f"{chunk.thinking[:400]}{COLOR_RESET}"
                    )
                if final_text and not assistant_stream_open:
                    if thinking_stream_open:
                        if COLOR_RESET:
                            print(COLOR_RESET, end="", flush=True)
                        print()
                    print(f"Assistant: {final_text}", end="", flush=True)
                memory.commit_assistant_response(
                    content=final_text,
                    model=chunk.model,
                    usage=chunk.usage,
                )
                memory.end_turn()
                print()
                break

        if not tool_turn:
            break

    return final_text


async def main() -> None:
    provider = AsyncOllamaProvider(
        model_id="qwen3-vl:4b",
        think=True,
    )
    embedder = OllamaEmbedder(model="embeddinggemma:latest")
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            context=ContextConfig(
                max_recent_messages=18,
                max_recent_tokens=9000,
                compression_enabled=True,
                compression_strategy="hierarchical",
                compress_after_turns=2,
                compress_after_tokens=1800,
                checkpoint_every_n_turns=2,
            ),
            memory=MemoryConfig(
                retrieve_top_k=6,
                relevance_threshold=0.1,
                include_graph_neighbors=True,
                max_neighbor_depth=1,
                retrieval_recent_ratio=0.5,
                retrieval_long_term_ratio=0.4,
                retrieval_tool_ratio=0.1,
            ),
            plan=PlanConfig(enabled=True, auto_infer=False, inject_in_context=True),
            tool_handler=ToolHandlerConfig(rewrite_failed_calls=True, soft_failure_message=True),
            tenant_id="demo-tenant",
            user_id="analyst-1",
            conversation_id="memory-insights-demo",
            session_id="session-001",
            persist_memories=True,
        ),
        embedder=embedder,
    )

    await memory.start()
    memory.set_plan(
        goal="Produce an EMEA Q3 snapshot with margin and risk commentary.",
        steps=[
            "Gather base metrics with tools",
            "Compute margin and growth interpretation",
            "Generate a concise recommendation",
        ],
    )

    prompts = [
        "Get EMEA Q3 sales metrics and compute the margin. Use tools where helpful.",
        "Compare that with EMEA Q4 and highlight risks if growth slows.",
        "Give me a final recommendation in 3 bullets and mention what should be monitored next.",
    ]

    try:
        async with AsyncAIClient(
            providers=[provider],
            urls=[],
            local_toolbox=toolbox,
            # Keep client prompt empty so memory controls system-level assembly.
            system_prompt="",
        ) as ai:
            for idx, prompt in enumerate(prompts):
                answer = await run_stream_turn(ai, memory, prompt)
                memory.update_plan_step(
                    step_index=min(idx, 2),
                    status="completed",
                    result_summary=answer[:180] if answer else "No answer captured",
                )

                # Explicitly store high-signal distilled memory for graph visibility.
                if answer:
                    await memory.add_memory(
                        content=f"Turn {idx + 1} distilled insight: {answer}",
                        memory_type="conversation",
                        metadata={"turn": idx + 1},
                    )

                if idx == 1:
                    memory.create_checkpoint("Post-comparison checkpoint")
                await asyncio.sleep(0.15)
                print_memory_insights(memory, turn_name=f"after turn {idx + 1}")

            query = "EMEA growth risk and margin changes"
            retrieved = await memory.search_memories(query, top_k=5)
            print(f"\n[Retrieval] query='{query}'")
            for mem in retrieved:
                print(f"- {mem.memory_type} | {mem.id} | {mem.compressed[:140]}")

            state_path = Path("examples/manual_loop/.agentic_memory_insights_state.json")
            await memory.save(str(state_path))
            restored = await AgenticMemory.load(str(state_path), embedder=embedder)
            restored_snapshot = restored.to_dict()
            print("\n[Persistence Round-Trip]")
            print(f"- Saved to: {state_path}")
            print(f"- Restored nodes: {len(restored_snapshot['graph']['nodes'])}")
            print(f"- Restored edges: {len(restored_snapshot['graph']['edges'])}")
            print(f"- Restored scope: {restored_snapshot['scope']}")
    finally:
        await memory.stop()


if __name__ == "__main__":
    asyncio.run(main())

