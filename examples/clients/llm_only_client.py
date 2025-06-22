import asyncio
from seqoria_agent import AIClient
from seqoria_agent.providers.openai_provider import OpenAIProvider


async def main():
    provider = OpenAIProvider(model_id="gpt-4o")
    async with AIClient(urls=[], providers=[provider]) as ai:
        while True:
            msg = input("You: ")
            if msg.lower() in {"exit", "quit"}:
                break
            res = await ai.chat(msg, history=[])
            print("Bot:", res["answer"])


if __name__ == "__main__":
    asyncio.run(main()) 