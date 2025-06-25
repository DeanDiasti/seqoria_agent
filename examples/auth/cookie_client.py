#!/usr/bin/env python3
import asyncio, jwt
from seqoria_agent.client import AIClient
from seqoria_agent.auth.providers import StaticCookieProvider
from seqoria_agent.providers import OpenAIProvider

SECRET = "change-me"
ALGO = "HS256"
BASE_URL = "http://localhost:8000/mcp/"


def make_token(user_id: str = "alice") -> str:
    return jwt.encode({"id": user_id}, SECRET, algorithm=ALGO)


async def main():
    token = make_token()
    async with AIClient(
        providers=[OpenAIProvider()],
        urls=[BASE_URL],
        auth_provider=StaticCookieProvider(token),
    ) as client:
        res = await client.call_mcp_tool("whoami", {})
        print(res[0].text)


if __name__ == "__main__":
    asyncio.run(main())
