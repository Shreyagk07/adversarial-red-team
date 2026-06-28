"""FastAPI application entrypoint.

Phase 0 exposes just enough to prove the backend boots and is wired to our
configuration seam:

  * ``GET /``        — a friendly root that names the service.
  * ``GET /health``  — a structured health check used by tests, CI, and the
                       deployment platform's liveness probe.

Run locally with::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from backend.api import router as api_router
from backend.config import Settings, get_settings
from storage.db import init_db


# --- Response schemas -------------------------------------------------------
# Declaring explicit response models gives us automatic validation, clean
# OpenAPI docs at /docs, and a stable contract for the dashboard to consume.
class HealthResponse(BaseModel):
    """Payload returned by the ``/health`` endpoint."""

    status: str
    app_name: str
    version: str
    environment: str
    llm_provider: str
    # Whether a key for the selected provider is present. We never return the
    # key itself — only a boolean — so health output is safe to log/expose.
    llm_key_configured: bool


def create_app() -> FastAPI:
    """Application factory.

    Using a factory (instead of a module-level ``app`` built inline) keeps
    construction testable and lets us inject settings/overrides later without
    import-time side effects.
    """
    settings: Settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Create the engine/session factory and tables on startup. Idempotent,
        # so it's safe across reloads and in tests.
        init_db()
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Automated robustness evaluation for AI agents. A Challenger "
            "agent probes a Task agent across robustness categories; a Judge "
            "scores each exchange and the system produces a robustness report."
        ),
        lifespan=lifespan,
    )

    # Mount the targets/runs/reports API.
    app.include_router(api_router)

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        """Human-friendly landing payload."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Structured liveness/readiness check.

        Reports which LLM provider is selected and whether its key is
        configured — without exposing the key — so we can diagnose
        misconfiguration at a glance.
        """
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
            llm_provider=settings.llm_provider,
            llm_key_configured=settings.active_provider_key_present,
        )

    return app


# The ASGI application object that uvicorn / deployment platforms import.
app = create_app()
