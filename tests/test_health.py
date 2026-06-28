"""Smoke tests for the Phase 0 backend.

These run without any API keys: they exercise the app factory and the
``/health`` contract using FastAPI's in-process test client (no server,
no network). This is the regression net every later phase builds on.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_returns_service_metadata() -> None:
    """The root endpoint should name the service and point at the docs."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "service" in body
    assert body["docs"] == "/docs"


def test_health_ok_and_contract() -> None:
    """``/health`` should report status ``ok`` and the full health contract."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    # Status is healthy...
    assert body["status"] == "ok"
    # ...and every field of the documented contract is present.
    for field in (
        "app_name",
        "version",
        "environment",
        "llm_provider",
        "llm_key_configured",
    ):
        assert field in body

    # The default provider is Groq and the key boolean is always a bool
    # (False in CI where no key is set).
    assert body["llm_provider"] in {"groq", "gemini"}
    assert isinstance(body["llm_key_configured"], bool)
