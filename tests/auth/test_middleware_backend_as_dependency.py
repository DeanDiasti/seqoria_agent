from fastapi import FastAPI
from fastapi.testclient import TestClient

from seqoria_agent.auth.backends import HeaderApiKeyBackend


def make_app():
    app = FastAPI()
    # Mount middleware directly on FastAPI application
    app.add_middleware(HeaderApiKeyBackend, key_header="X-API-Key", validator="demo")

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    return app


def test_middleware_backend_http_unauth():
    app = make_app()
    cli = TestClient(app)

    res = cli.get("/ping")
    assert res.status_code == 401


def test_middleware_backend_http_auth():
    app = make_app()
    cli = TestClient(app)

    res = cli.get("/ping", headers={"X-API-Key": "demo"})
    assert res.status_code == 200
    assert res.json() == {"pong": True} 