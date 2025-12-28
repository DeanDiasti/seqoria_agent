import asyncio
from seqoria_agent import AIClient
from seqoria_agent.llm_providers.openai_provider import OpenAIProvider
from seqoria_agent.models.llm_models import AssistantMessage, ToolMessages

async def main() -> None:
    provider = OpenAIProvider(model_id="gpt-4o")
    async with AIClient(urls=[], providers=[provider]) as ai:
        while True:
            user_query = input("You:\n")
            if user_query.lower() in {"exit", "quit"}:  
                break
            async for chunk in ai.chat(user_query=user_query, history=[]):
                if isinstance(chunk, AssistantMessage):
                    print(f"Bot:\n{chunk.content}")
                elif isinstance(chunk, ToolMessages):
                    print("tool call\n" * (len(chunk) // 2))


if __name__ == "__main__":
    asyncio.run(main()) 