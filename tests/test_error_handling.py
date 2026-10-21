import anthropic
import httpx2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import anthropic_auth_error_handler


def _auth_error() -> anthropic.AuthenticationError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(401, request=request)
    return anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)


def test_anthropic_auth_error_is_a_clear_502_and_logged(caplog):
    app = FastAPI()
    app.add_exception_handler(anthropic.AuthenticationError, anthropic_auth_error_handler)

    @app.get("/boom")
    def boom():
        raise _auth_error()

    with caplog.at_level("ERROR", logger="app.main"):
        resp = TestClient(app).get("/boom")

    assert resp.status_code == 502
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]
    assert "rejected the configured key" in caplog.text
