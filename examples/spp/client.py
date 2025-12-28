"""Minimal demo for Server-Populated Parameters v2.

Run with:
    python examples/param_demo/main.py
Assumes a local FastXMCP server exposing a tool `hello` that requires
`user_id` as a server-populated parameter.
"""

import asyncio
import os
from seqoria_agent import AIClient
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.llm_providers import OpenAIProvider
from dotenv import load_dotenv
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.ai_client.base import ToolCallRequest
from pydantic import BaseModel

load_dotenv()

class Address(BaseModel):
    street: str
    city: str
    country: str


tb = LocalToolbox()

def get_user_id(tenant_id: str | None = None) -> str:
    return "u-42" if tenant_id == "acme" else "anonymous"


def get_address() -> Address:
    """Return a demo Address instance."""
    return Address(street="1 Infinite Loop", city="Cupertino", country="USA")

def get_user_secret_value(user_id: str) -> str:
    return "secret" if user_id == "u-42" else "wrong"

@tb.tool(server_populated=["user_id", "tenant_id"])
async def hello(user_id: str, tenant_id: str) -> dict:
    """Return a greeting that proves SPP injection worked with structured data."""

    return {"msg": f"hello {user_id} from {tenant_id}"}

@tb.tool(server_populated=["address", "user_id"])
async def hello_address(address: Address, user_id: str) -> dict:
    """Return a greeting that proves SPP injection worked with structured data."""
    if user_id == "u-42":
        return {"msg": f"{address.street} {address.city} {address.country}"}
    else:
        return {"msg": f"No address for {user_id}"}
    
@tb.tool(server_populated=["user_secret"])
async def get_user_secret(user_secret: str) -> dict:
    """Return a user secret."""
    if user_secret == "secret":
        return {"msg": f"user secret: {user_secret}"}
    else:
        return {"msg": f"No user secret"}

# ------------------------------------------------------------------ parameter registry
registry = ParameterRegistry(mapping={"user_id": get_user_id, 
                                      "tenant_id": "acme", 
                                      "address": get_address,
                                      "user_secret": get_user_secret_value})

async def main() -> None:
    async with AIClient(providers=[OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))],
                        param_registry=registry,
                        local_toolbox=tb) as client:
        res = await client.run_tool_calls(tool_calls=[ToolCallRequest(id="0",
                                                                      role="tool_call",
                                                                      type="function_call",
                                                                      name="hello",
                                                                      arguments={})],
                                          context={"tenant_id": "acme"},)
        print(res[0].result, "\n\n")

        res = await client.run_tool_calls(tool_calls=[ToolCallRequest(id="1",
                                                                      role="tool_call",
                                                                      type="function_call",
                                                                      name="hello",
                                                                      arguments={})],
                                          context={"tenant_id": "boss"},)
        print(res[0].result, "\n\n")

        res = await client.run_tool_calls(tool_calls=[ToolCallRequest(id="2",
                                                                      role="tool_call",
                                                                      type="function_call",
                                                                      name="hello_address",
                                                                      arguments={})],
                                          context={"tenant_id": "acme"},)
        print(res[0].result, "\n\n")

        res = await client.run_tool_calls(tool_calls=[ToolCallRequest(id="3",
                                                                      role="tool_call",
                                                                      type="function_call",
                                                                      name="hello_address",
                                                                      arguments={})],
                                          context={"tenant_id": "boss"},)
        print(res[0].result, "\n\n")

        res = await client.run_tool_calls(tool_calls=[ToolCallRequest(id="4",
                                                                      role="tool_call",
                                                                      type="function_call",
                                                                      name="get_user_secret",
                                                                      arguments={})],
                                          context={"user_secret": "secret"},)
        print(res[0].result, "\n\n")

        res = await client.run_tool_calls(tool_calls=[ToolCallRequest(id="5",
                                                                      role="tool_call",
                                                                      type="function_call",
                                                                      name="get_user_secret",
                                                                      arguments={})],
                                          context={"user_id": "u-42"},)
        print(res[0].result, "\n\n")



if __name__ == "__main__":
    asyncio.run(main()) 