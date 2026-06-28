"""Tests for the optional Langfuse tracing layer (Phase 9).

Offline and key-free: we verify that with no credentials, tracing degrades to
a clean no-op (None handler, empty callbacks, no-op context manager), and that
the env-export helper maps settings to the SDK's expected variables.
"""

from __future__ import annotations

import os

import agents.observability as obs
from agents.observability import _ensure_env, observe_run, tracing_callbacks
from backend.config import Settings


def _reset_cache() -> None:
    """Reset the module-level handler cache for a deterministic test."""
    obs._handler = None
    obs._handler_initialized = False


def test_tracing_disabled_returns_empty_callbacks() -> None:
    _reset_cache()
    settings = Settings(langfuse_public_key=None, langfuse_secret_key=None)
    assert obs.get_langfuse_handler(settings) is None
    assert tracing_callbacks(settings) == []


def test_observe_run_is_noop_when_disabled() -> None:
    settings = Settings(langfuse_public_key=None, langfuse_secret_key=None)
    # Should simply run the block without error and without Langfuse.
    ran = False
    with observe_run("test", settings):
        ran = True
    assert ran is True


def test_langfuse_enabled_property() -> None:
    assert Settings(langfuse_public_key="pk", langfuse_secret_key="sk").langfuse_enabled
    assert not Settings(langfuse_public_key="pk").langfuse_enabled
    assert not Settings().langfuse_enabled


def test_ensure_env_exports_credentials(monkeypatch) -> None:  # noqa: ANN001
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(
        langfuse_public_key="pk-lf-x",
        langfuse_secret_key="sk-lf-y",
        langfuse_host="https://example.langfuse.com",
    )
    _ensure_env(settings)
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-lf-x"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-lf-y"
    assert os.environ["LANGFUSE_HOST"] == "https://example.langfuse.com"
