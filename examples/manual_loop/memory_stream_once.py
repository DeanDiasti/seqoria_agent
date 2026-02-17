from __future__ import annotations

"""Manual memory-first loop using llm_stream_once().

Run:
    python examples/manual_loop/memory_stream_once.py
"""

import asyncio

from seqoria_agent import AsyncOllamaProvider
from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk, ToolCallRequestList
from seqoria_agent.toolbox import LocalToolbox


box = LocalToolbox()


@box.tool(description="Echoes the message back")
def echo(message: str) -> str:
    return message


@box.tool(description="Reverses the input string")
def reverse(message: str) -> str:
    return message[::-1]


@box.tool(description="Counts words in the input string")
def word_count(message: str) -> int:
    return len(message.split())


async def main() -> None:
    # Use Ollama provider with the requested model. If using Ollama Cloud,
    # you can set OLLAMA_API_KEY in the environment; for a local Ollama server,
    # set OLLAMA_HOST (defaults to http://localhost:11434) if needed.
    provider = AsyncOllamaProvider(
        model_id="qwen3-vl:4b-instruct",
    )
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            session_id="memory-stream-demo",
            conversation_id="memory-stream-loop",
        )
    )

    async with AsyncAIClient(providers=[provider], urls=[], local_toolbox=box) as ai:
        while True:
            user_query = input("You: ").strip()
            if not user_query:
                break

            llm_history = memory.begin_turn(user_query, retrieval_query=user_query)

            for _ in range(5):
                tool_turn = False
                streamed_parts: list[str] = []
                async for chunk in ai.llm_stream_once(history=llm_history):
                    if isinstance(chunk, ChatChunk):
                        print(chunk.content, end="", flush=True)
                        streamed_parts.append(chunk.content)
                        continue
                    if isinstance(chunk, ToolCallRequestList):
                        # Show tool request(s) in the CLI so the user knows what's happening.
                        print("\n\n=== Tool call requested ===")
                        for tc in chunk.tool_calls:
                            print(f"- Tool: {tc.name}  id: {tc.id}  args: {tc.arguments}")

                        tool_results = await ai.run_tool_calls(tool_calls=chunk.tool_calls)

                        print("=== Tool results ===")
                        for tr in tool_results:
                            print(
                                f"- Tool: {tr.name}  id: {tr.tool_call_id}  result: {tr.result}"
                            )

                        memory.record_tool_cycle(chunk.tool_calls, tool_results)
                        llm_history = memory.messages_for_llm(query=user_query)
                        tool_turn = True
                        print("=== Returning to LLM ===\n")
                        break

                    if isinstance(chunk, AssistantMessage):
                        final_text = chunk.content or "".join(streamed_parts)
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


if __name__ == "__main__":
    asyncio.run(main())

