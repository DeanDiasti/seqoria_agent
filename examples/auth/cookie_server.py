#!/usr/bin/env python3
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import BearerCookieBackend

SECRET = "change-me"
ALGO = "HS256"

app = FastXMCP(auth_backend=BearerCookieBackend(secret=SECRET, algorithm=ALGO))


@app.tool(server_populated=["user_id"])
async def whoami(user_id: str):  # noqa: D401
    """Return the caller's user_id extracted from the JWT."""
    return {"user_id": user_id}


if __name__ == "__main__":
    app.run()
