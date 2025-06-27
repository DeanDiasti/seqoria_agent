import asyncio
from seqoria_agent import AIClient
from seqoria_agent.providers.openai_provider import OpenAIProvider


async def main() -> None:
    provider = OpenAIProvider(model_id="gpt-4o")
    async with AIClient(urls=[], providers=[provider]) as ai:
        while True:
            user_query = input("You:\n")
            if user_query.lower() in {"exit", "quit"}:  
                break
            res = await ai.chat(user_query=user_query, history=[])
            print("Bot\n:", res["answer"])


if __name__ == "__main__":
    asyncio.run(main()) 