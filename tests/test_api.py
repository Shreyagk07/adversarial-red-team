"""API tests for targets, runs, and reports (Phase 7).

Offline: we point DATABASE_URL at a temp SQLite file (via env + cleared
settings cache) so the real app — lifespan, router, repository — runs against an
isolated database. No API key is set, which lets us assert the evaluate
endpoint's clear 503 fail-fast behavior without launching a real run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Isolate the database to a temp file and ensure no provider key is set.
    db_url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Clear the cached settings so the app reads the temp DATABASE_URL. We must
    # NOT reload storage.db here: that would create a second `Base` class that
    # storage.models no longer shares, leaving create_all with no tables.
    import backend.config as config

    config.get_settings.cache_clear()

    from backend.main import create_app

    app = create_app()
    with TestClient(app) as c:  # triggers lifespan -> init_db() on temp DB
        yield c

    config.get_settings.cache_clear()


def test_health_still_ok(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_register_and_fetch_target(client: TestClient) -> None:
    resp = client.post("/targets", json={"name": "Demo", "description": "d"})
    assert resp.status_code == 201
    target = resp.json()
    assert target["id"] and target["name"] == "Demo"

    # It appears in the list and can be fetched by id.
    assert any(t["id"] == target["id"] for t in client.get("/targets").json())
    assert client.get(f"/targets/{target['id']}").json()["name"] == "Demo"


def test_fetch_missing_target_404(client: TestClient) -> None:
    assert client.get("/targets/nope").status_code == 404


def test_evaluate_without_key_returns_503(client: TestClient) -> None:
    target_id = client.post("/targets", json={"name": "Demo"}).json()["id"]
    resp = client.post(f"/targets/{target_id}/evaluate", json={"tests_per_category": 2})
    # No key configured -> clear, actionable 503 (run is not started).
    assert resp.status_code == 503
    assert "No API key" in resp.json()["detail"]


def test_evaluate_missing_target_404(client: TestClient) -> None:
    resp = client.post("/targets/nope/evaluate", json={})
    assert resp.status_code == 404


def test_report_for_unknown_run_404(client: TestClient) -> None:
    assert client.get("/runs/nope/report").status_code == 404


def _seed_completed_run(run_id: str, target_id: str, created_at: str, overall: float):
    """Insert a completed run with a minimal report, via the active temp DB."""
    from agents.schemas import CategorySummary, RobustnessReport
    from storage.db import session_scope
    from storage.models import Run, Target

    report = RobustnessReport(
        target_name=run_id,
        target_description="d",
        created_at=created_at,
        overall_score=overall,
        overall_pass_rate=overall / 100,
        total_tests=5,
        total_passed=0,
        total_failed=0,
        category_summaries=[
            CategorySummary(
                category_id="ambiguity",
                name="Ambiguity handling",
                score=overall,
                num_tests=5,
                num_passed=0,
                num_failed=5,
                severity_counts={"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0},
            )
        ],
    )
    with session_scope() as s:
        if s.get(Target, target_id) is None:
            s.add(Target(id=target_id, name="demo", created_at=created_at))
        s.add(
            Run(
                id=run_id,
                target_id=target_id,
                status="completed",
                tests_per_category=2,
                overall_score=overall,
                created_at=created_at,
                report_json=report.model_dump_json(),
            )
        )


def test_compare_two_completed_runs(client: TestClient) -> None:
    _seed_completed_run("before1", "t1", "2026-06-28T00:00:00+00:00", overall=40.0)
    _seed_completed_run("after1", "t1", "2026-06-28T01:00:00+00:00", overall=80.0)

    resp = client.get("/compare", params={"before": "before1", "after": "after1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_delta"] == 40.0
    assert body["num_improved"] == 1


def test_compare_missing_run_returns_409(client: TestClient) -> None:
    resp = client.get("/compare", params={"before": "nope", "after": "nope2"})
    assert resp.status_code == 409
