"""Langfuse observability — trace every agent LLM call.

This is the *one* place tracing is wired. It's entirely optional and fails
open: if Langfuse isn't configured (no keys) or the package isn't installed,
every function here degrades to a no-op so the system runs exactly as before.

How it's used:
  * Each agent (Task/Challenger/Judge) attaches :func:`tracing_callbacks` to its
    LangChain runnable, so individual calls are captured with tokens/latency.
  * Long-running jobs wrap their work in :func:`observe_run`, which opens a
    parent span so all of a suite's calls nest under one trace, then flushes.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from backend.config import Settings, get_settings

# Cache the handler per-process so we don't spin up a new one per agent.
_handler = None
_handler_initialized = False


def _ensure_env(settings: Settings) -> None:
    """Export Langfuse credentials to the env vars the SDK reads."""
    if settings.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if settings.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    # Newer SDKs read LANGFUSE_HOST; set the legacy name too, just in case.
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host


def get_langfuse_handler(settings: Settings | None = None):
    """Return a cached Langfuse CallbackHandler, or None if unavailable."""
    global _handler, _handler_initialized
    if _handler_initialized:
        return _handler

    settings = settings or get_settings()
    _handler_initialized = True

    if not settings.langfuse_enabled:
        _handler = None
        return None

    _ensure_env(settings)
    try:
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
    except Exception:
        # Package missing OR handler construction failed (bad keys, etc.).
        # Tracing must never break the app, so we fail open and disable it.
        _handler = None
    return _handler


def tracing_callbacks(settings: Settings | None = None) -> list:
    """Return ``[handler]`` for LangChain config, or ``[]`` when disabled."""
    handler = get_langfuse_handler(settings)
    return [handler] if handler is not None else []


@contextmanager
def observe_run(name: str, settings: Settings | None = None) -> Iterator[None]:
    """Group all traced calls within the block under one parent span.

    Fail-open by design: any problem setting up or tearing down tracing is
    swallowed so the wrapped work always runs. We only guard the Langfuse
    setup/teardown — never the user's block — so real errors still propagate.
    Flushes queued traces on exit so short-lived runs don't lose data.
    """
    settings = settings or get_settings()

    client = None
    span = None
    if settings.langfuse_enabled:
        try:
            _ensure_env(settings)
            from langfuse import get_client

            client = get_client()
            # NB: langfuse 4.x uses start_as_current_observation, not
            # start_as_current_span. as_type='span' is the default.
            span = client.start_as_current_observation(name=name, as_type="span")
            span.__enter__()
        except Exception:
            span = None  # tracing unavailable; carry on without it

    try:
        yield
    finally:
        if span is not None:
            try:
                span.__exit__(None, None, None)
            except Exception:
                pass
        if client is not None:
            try:
                client.flush()
            except Exception:
                pass
