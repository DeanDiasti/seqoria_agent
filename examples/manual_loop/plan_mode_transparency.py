from __future__ import annotations

"""Plan-mode transparency demo using AgenticMemory + manual loop.

This demo uses a real Ollama model:
    qwen3-vl:4b-instruct

What it shows:
- A multi-step plan inferred automatically by AgenticMemory
- The plan block injected into the LLM prompt before each call
- Tool-call cycles in a manual loop (`llm_generate_once` + `run_tool_calls`)
- Step-by-step auto-progress updates driven by execution evidence

Run:
    uv run examples/manual_loop/plan_mode_transparency.py

Prerequisites:
    ollama pull qwen3-vl:4b-instruct
    ollama serve
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seqoria_agent import AsyncOllamaProvider
from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig, Plan, PlanConfig
from seqoria_agent.models.chunk import AssistantMessage, ToolCallError, ToolCallRequestList, ToolCallResult
from seqoria_agent.toolbox import LocalToolbox


toolbox = LocalToolbox()

class _ArtifactIO:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.action_log_path = self.base_dir / "action_log.jsonl"
        self.last_walkthrough_path: Path | None = None

    def append_action_log(self, tool_name: str, payload: dict[str, Any], result: str) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "payload": payload,
            "result_preview": result[:280],
        }
        with self.action_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_walkthrough(self, markdown: str, filename: str) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name.replace(" ", "_")
        if not safe_name.endswith(".md"):
            safe_name = f"{safe_name}.md"
        target = self.base_dir / safe_name
        target.write_text(markdown.strip() + "\n", encoding="utf-8")
        self.last_walkthrough_path = target
        return target


artifact_io = _ArtifactIO(Path("examples/manual_loop/.plan_mode_artifacts"))


@toolbox.tool(description="Collect concise requirements for a topic.")
def collect_requirements(topic: str) -> str:
    result = (
        f"Requirements for '{topic}': show active plan injection, show tool cycle, "
        "show plan status changes after each assistant step."
    )
    artifact_io.append_action_log("collect_requirements", {"topic": topic}, result)
    return result


@toolbox.tool(description="Draft a short design based on requirements text.")
def draft_design(requirements: str) -> str:
    result = (
        "Design: rely on memory-level auto planning in begin_turn/record_tool_cycle/commit_assistant_response, "
        "print active plan blocks from messages_for_llm(), and execute tool loop for concrete evidence. "
        f"Requirements seen: {requirements[:100]}"
    )
    artifact_io.append_action_log("draft_design", {"requirements": requirements[:180]}, result)
    return result


@toolbox.tool(
    description=(
        "Write the final walkthrough to disk and return the saved file path. "
        "Use this to complete execution, not just planning."
    )
)
def save_walkthrough(markdown: str, filename: str = "plan_mode_walkthrough.md") -> str:
    target = artifact_io.save_walkthrough(markdown, filename)
    result = f"Saved walkthrough to {target}"
    artifact_io.append_action_log(
        "save_walkthrough",
        {"filename": target.name, "chars_written": len(markdown)},
        result,
    )
    return result


def _render_plan(plan: Plan | None) -> str:
    if plan is None:
        return "(no active plan)"
    lines = [f"Goal: {plan.goal}", plan.progress_summary]
    for idx, step in enumerate(plan.steps, start=1):
        marker = "->" if idx - 1 == plan.current_step_index else "  "
        lines.append(f"{marker} {idx}. [{step.status}] {step.description}")
        if step.result_summary:
            lines.append(f"   result: {step.result_summary}")
    return "\n".join(lines)


def _extract_plan_block(history: list[dict[str, Any]]) -> str | None:
    system_msg = next((m for m in history if m.get("role") == "system"), None)
    if not system_msg:
        return None
    content = str(system_msg.get("content", ""))
    if "<active_plan>" not in content:
        return None
    start = content.find("Goal:")
    if start < 0:
        return None
    tail = content[start:]
    end_markers = [
        marker
        for marker in ("\n<tool_guidance>", "\n<earlier_conversation>", "\n<relevant_memories>")
        if marker in tail
    ]
    if end_markers:
        end_idx = min(tail.find(marker) for marker in end_markers)
        tail = tail[:end_idx]
    return tail.strip()


def _print_plan_transparency(history: list[dict[str, Any]]) -> None:
    block = _extract_plan_block(history)
    print("\n[plan-mode transparency]")
    print(f"- active plan injected: {bool(block)}")
    if not block:
        print("- plan block: (not found)")
        return
    print("- plan block sent to the LLM:")
    for line in block.splitlines():
        print(f"  {line}")


def _tool_result_text(result: ToolCallResult | ToolCallError) -> str:
    if isinstance(result, ToolCallError):
        return str(result.error)
    return str(result.result)


def _plan_is_complete(plan: Plan | None) -> bool:
    if plan is None or not plan.steps:
        return False
    return all(step.status in {"completed", "skipped"} for step in plan.steps)


async def main() -> None:
    provider = AsyncOllamaProvider(model_id="qwen3-vl:4b-instruct")
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            plan=PlanConfig(
                enabled=True,
                auto_infer=True,
                inject_in_context=True,
                update_on_tool_success=True,
                assistant_progress_enabled=True,
                replan_on_blocked=True,
                complexity_min_words=12,
                max_inferred_steps=5,
            ),
            session_id="plan-mode-demo-session",
            conversation_id="plan-mode-demo-conversation",
        )
    )
    seed_prompt = (
        "Create an implementation brief for improving agent memory orchestration in this project. "
        "Use available tools to gather requirements, draft the technical design details, "
        "and save a final markdown brief to disk."
    )
    followup_prompt = (
        "Continue from current progress. If the deliverable is incomplete, take the next concrete action and finish it."
    )

    async with AsyncAIClient(
        providers=[provider],
        urls=[],
        local_toolbox=toolbox,
        system_prompt="",
    ) as ai:
        prompt = seed_prompt
        for turn_index in range(6):
            print(f"\n\n=== Turn {turn_index + 1} ===")
            print(f"You: {prompt}")
            llm_history = memory.begin_turn(prompt, retrieval_query=prompt)
            _print_plan_transparency(llm_history)

            for _ in range(8):
                response = await ai.llm_generate_once(history=llm_history, use_tools=True)

                if isinstance(response, ToolCallRequestList):
                    print("\n[tool-call]")
                    for call in response.tool_calls:
                        print(f"- {call.name} id={call.id} args={call.arguments}")

                    tool_results = await ai.run_tool_calls(response.tool_calls)
                    print("[tool-result]")
                    for result in tool_results:
                        result_text = _tool_result_text(result)
                        print(f"- {result.name}: {result_text[:140]}")

                    memory.record_tool_cycle(response.tool_calls, tool_results)
                    llm_history = memory.messages_for_llm(query=prompt)
                    _print_plan_transparency(llm_history)
                    continue

                if isinstance(response, AssistantMessage):
                    assistant_text = response.content or ""
                    print(f"\nAssistant: {assistant_text}")
                    memory.commit_assistant_response(
                        content=assistant_text,
                        model=response.model,
                        usage=response.usage,
                    )
                    memory.end_turn()
                    print("\n[plan after auto-update]")
                    print(_render_plan(memory.current_plan))
                    break

            if _plan_is_complete(memory.current_plan):
                break
            prompt = followup_prompt

    print("\n=== Final Plan State ===")
    print(_render_plan(memory.current_plan))
    if artifact_io.last_walkthrough_path and artifact_io.last_walkthrough_path.exists():
        print(f"\n[artifact] Walkthrough file: {artifact_io.last_walkthrough_path}")
    if artifact_io.action_log_path.exists():
        print(f"[artifact] Action log: {artifact_io.action_log_path}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
