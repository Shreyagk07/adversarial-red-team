"""Tests for the persistence layer (Phase 7).

Offline: each test builds its own SQLAlchemy engine against a temp SQLite file,
so there's no shared state and no network. We verify target CRUD, the run
lifecycle (running -> completed / failed), normalized child rows, and lossless
report reconstruction.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import sessionmaker

from agents.base import AgentResponse
from agents.schemas import (
    CategoryScore,
    Exchange,
    LoopResult,
    Mitigation,
    RobustnessReport,
    SuiteResult,
    TestCase,
    Verdict,
)
from storage import repository as repo
from storage.db import Base, make_engine


def _make_session(tmp_path: Path):
    """Build an isolated SQLite session factory for a test."""
    import storage.models  # noqa: F401 - register tables on Base

    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _sample_suite_and_report() -> tuple[SuiteResult, RobustnessReport]:
    tc = TestCase(
        id="ambiguity-001",
        category_id="ambiguity",
        probe_type="missing referent",
        prompt="Is it better?",
        rationale="no referent",
        expected_behavior="ask what 'it' means",
    )
    resp = AgentResponse(text="Yes!", provider="local", model="echo", latency_ms=1.0)
    ex = Exchange(
        test_case=tc,
        response=resp,
        verdict=Verdict(passed=False, severity="high", reason="guessed"),
    )
    result = LoopResult(
        target_name="demo",
        target_description="d",
        category_id="ambiguity",
        exchanges=[ex],
        category_score=CategoryScore.from_exchanges("ambiguity", [ex]),
    )
    suite = SuiteResult.from_results("demo", "d", [result])
    report = RobustnessReport(
        target_name="demo",
        target_description="d",
        created_at="2026-06-28T00:00:00+00:00",
        overall_score=suite.overall_score,
        overall_pass_rate=suite.overall_pass_rate,
        total_tests=suite.total_tests,
        total_passed=suite.total_passed,
        total_failed=suite.total_failed,
        worst_failures=[ex],
        mitigations=[
            Mitigation(
                category_id="ambiguity",
                issue="guesses",
                suggestion="ask to clarify",
                priority="high",
            )
        ],
    )
    return suite, report


# --- targets -----------------------------------------------------------------
def test_target_crud(tmp_path: Path) -> None:
    Session = _make_session(tmp_path)
    with Session() as s:
        t = repo.create_target(s, name="My Agent", description="desc", provider="groq")
        assert t.id and t.created_at
        fetched = repo.get_target(s, t.id)
        assert fetched is not None and fetched.name == "My Agent"
        assert [x.id for x in repo.list_targets(s)] == [t.id]


# --- run lifecycle -----------------------------------------------------------
def test_run_completes_and_persists_children(tmp_path: Path) -> None:
    Session = _make_session(tmp_path)
    suite, report = _sample_suite_and_report()

    with Session() as s:
        target = repo.create_target(s, name="demo")
        run = repo.create_run(s, target_id=target.id, tests_per_category=1)
        assert run.status == "running"

        completed = repo.complete_run(s, run.id, suite=suite, report=report)
        assert completed.status == "completed"
        assert completed.overall_score == suite.overall_score
        assert completed.total_failed == 1

        # Normalized children were written.
        assert len(completed.category_scores) == 1
        assert len(completed.case_results) == 1
        assert completed.case_results[0].passed is False
        assert len(completed.mitigations) == 1

        # Report reconstructs losslessly from the stored JSON.
        reloaded = repo.get_run_report(s, run.id)
        assert reloaded is not None
        assert reloaded.mitigations[0].suggestion == "ask to clarify"
        assert reloaded.worst_failures[0].test_case.id == "ambiguity-001"


def test_run_can_be_marked_failed(tmp_path: Path) -> None:
    Session = _make_session(tmp_path)
    with Session() as s:
        target = repo.create_target(s, name="demo")
        run = repo.create_run(s, target_id=target.id, tests_per_category=1)
        repo.mark_run_failed(s, run.id, "boom")

        reloaded = repo.get_run(s, run.id)
        assert reloaded is not None
        assert reloaded.status == "failed"
        assert reloaded.error == "boom"
        assert repo.get_run_report(s, run.id) is None


def test_list_runs_filters_by_target(tmp_path: Path) -> None:
    Session = _make_session(tmp_path)
    with Session() as s:
        a = repo.create_target(s, name="A")
        b = repo.create_target(s, name="B")
        run_a = repo.create_run(s, target_id=a.id, tests_per_category=1)
        repo.create_run(s, target_id=b.id, tests_per_category=1)

        a_runs = repo.list_runs(s, target_id=a.id)
        assert [r.id for r in a_runs] == [run_a.id]
        assert len(repo.list_runs(s)) == 2
