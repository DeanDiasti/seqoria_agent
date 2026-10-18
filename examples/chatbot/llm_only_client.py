import asyncio
from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import AsyncOllamaProvider
from seqoria_agent.models.chunk import AssistantMessage, ToolMessages


async def main() -> None:
    provider = AsyncOllamaProvider(model_id="gpt-oss:20b")
    async with provider, AsyncAIClient(mcp_servers=[], providers=[provider]) as ai:
        history = []
        while True:
            user_query = input("You:\n")
            if user_query.lower() in {"exit", "quit"}:
                break
            async for chunk in ai.chat(user_query=user_query, history=history):
                if isinstance(chunk, AssistantMessage):
                    print(f"Bot:\n{chunk.content}")
                elif isinstance(chunk, ToolMessages):
                    print("tool call\n" * (len(chunk) // 2))


if __name__ == "__main__":
    asyncio.run(main())
