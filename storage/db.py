"""Database engine, session management, and schema bootstrap.

We use SQLAlchemy 2.0 so the same ORM models run on SQLite (dev, zero setup)
and Postgres (prod, e.g. Neon) — switching is purely a matter of pointing
``DATABASE_URL`` at a different database. That is the only seam: no model or
query code is database-specific.

The engine/session factory are initialized lazily via :func:`init_db` (called
from the FastAPI lifespan, and from tests against a temp database), avoiding
import-time side effects.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# Module-level engine/session factory, set by init_db().
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def make_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for ``url`` (or the configured default).

    SQLite needs ``check_same_thread=False`` so the connection can be used
    across FastAPI's threadpool / background tasks; other backends ignore it.
    """
    url = url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


def init_db(url: str | None = None) -> Engine:
    """Initialize the engine + session factory and create tables.

    Idempotent enough for app startup and tests. Importing ``models`` here
    ensures every table is registered on ``Base.metadata`` before create_all.
    """
    global _engine, _SessionLocal

    _engine = make_engine(url)
    # Import models for their side effect of registering tables on Base.
    from storage import models  # noqa: F401

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Return the active session factory, initializing with defaults if needed."""
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope: commit on success, rollback on error.

    Used by background jobs and scripts that manage their own session.
    """
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Commits are the caller/repository's responsibility; we just ensure the
    session is always closed.
    """
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
