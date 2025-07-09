import asyncio

from seqoria_agent import AIClient
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.models.chunk import AssistantMessage, ToolMessages
from seqoria_agent.providers.openai_provider import OpenAIProvider


toolbox = LocalToolbox()


@toolbox.tool()
def add(a: int, b: int) -> int:
    """Return a + b (local tool)."""
    return int(a) + int(b)


@toolbox.tool()
def shout(msg: str):
    """Simple shouting tool returns uppercase version of the message."""
    return msg.upper()


async def main():
    provider = OpenAIProvider(model_id="gpt-4o")
    async with AIClient(urls=[], providers=[provider], local_toolbox=toolbox) as ai:
        async for chunk in ai.chat(user_query="What is 2+3?", history=[]):
            if isinstance(chunk, AssistantMessage):
                print(f"Answer\n: {chunk.content}")
            elif isinstance(chunk, ToolMessages):
                print("tool call\n" * (len(chunk) // 2))


if __name__ == "__main__":
    asyncio.run(main()) 