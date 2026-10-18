"""Resolve trusted local parameters without a model or remote server.

Run: python examples/spp/client.py

The application supplies authenticated request context. ParameterRegistry does
not infer dependencies between providers; each provider receives that context.
Remote MCP identity must instead be resolved by the authenticated server.
"""

import asyncio

from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.toolbox import LocalToolbox


toolbox = LocalToolbox()


@toolbox.tool(server_populated=["user_id", "tenant_id"])
async def hello(name: str, user_id: str, tenant_id: str) -> str:
    return f"Hello {name}: user={user_id}, tenant={tenant_id}"


def resolve_user(authenticated_user_id: str) -> str:
    return authenticated_user_id


registry = ParameterRegistry({"user_id": resolve_user})


async def main() -> None:
    # In a web application these values come from authentication middleware.
    for context in [
        {"authenticated_user_id": "user-a", "tenant_id": "tenant-a"},
        {"authenticated_user_id": "user-b", "tenant_id": "tenant-b"},
    ]:
        user_id = await registry.resolve("user_id", context=context)
        result = await toolbox.call_tool(
            "hello",
            {"name": "Alice"},
            server_args={"user_id": user_id, "tenant_id": context["tenant_id"]},
        )
        print(result[0].text)


if __name__ == "__main__":
    asyncio.run(main())
