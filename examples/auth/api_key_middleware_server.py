from fastapi import FastAPI
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import HeaderApiKeyBackend

API_KEY = "demo"

app = FastXMCP(title="APIKeyMiddlewareExample")
# Mount middleware on *both* regular FastAPI routes and XMCP tools
app.add_middleware(HeaderApiKeyBackend, key_header="X-API-Key", validator=API_KEY)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.tool()
async def ping():
    """Simple echo tool to verify authentication."""
    return "pong"


if __name__ == "__main__":  # pragma: no cover – manual run
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000) 