"""Repository layer — the only module that talks to the ORM.

Keeping all persistence logic here (rather than sprinkling queries through the
API/agents) means the rest of the app deals in domain objects and never sees a
SQLAlchemy session detail. If we ever change databases or ORMs, the blast
radius is this file.

Functions take an explicit :class:`Session` so callers control transaction
boundaries (request scope, background job, or test).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.schemas import RobustnessReport, SuiteResult
from storage.models import CaseResult, CategoryScoreRow, MitigationRow, Run, Target


def _new_id() -> str:
    """Short, URL-safe unique id."""
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# --- Targets -----------------------------------------------------------------
def create_target(
    session: Session,
    *,
    name: str,
    description: str = "",
    system_prompt: str = "",
    provider: str = "groq",
    model: str | None = None,
    temperature: float = 0.3,
) -> Target:
    """Insert and return a new target."""
    target = Target(
        id=_new_id(),
        name=name,
        description=description,
        system_prompt=system_prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        created_at=_now_iso(),
    )
    session.add(target)
    session.commit()
    session.refresh(target)
    return target


def get_target(session: Session, target_id: str) -> Target | None:
    return session.get(Target, target_id)


def list_targets(session: Session) -> list[Target]:
    stmt = select(Target).order_by(Target.created_at.desc())
    return list(session.scalars(stmt))


# --- Runs --------------------------------------------------------------------
def create_run(
    session: Session, *, target_id: str, tests_per_category: int
) -> Run:
    """Create a run in 'running' status (before the evaluation starts)."""
    run = Run(
        id=_new_id(),
        target_id=target_id,
        status="running",
        tests_per_category=tests_per_category,
        created_at=_now_iso(),
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def mark_run_failed(session: Session, run_id: str, error: str) -> None:
    """Record that a run failed, with the error message."""
    run = session.get(Run, run_id)
    if run is None:
        return
    run.status = "failed"
    run.error = error
    session.commit()


def complete_run(
    session: Session,
    run_id: str,
    *,
    suite: SuiteResult,
    report: RobustnessReport,
) -> Run:
    """Persist a finished evaluation: headline metrics, normalized rows, JSON.

    Stores the full report JSON for lossless reconstruction and writes the
    normalized child rows (category scores, per-case results, mitigations) for
    querying and drill-down.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run {run_id!r} not found")

    # Headline metrics + full report blob.
    run.status = "completed"
    run.overall_score = suite.overall_score
    run.overall_pass_rate = suite.overall_pass_rate
    run.total_tests = suite.total_tests
    run.total_passed = suite.total_passed
    run.total_failed = suite.total_failed
    run.report_json = report.model_dump_json()

    # Normalized children.
    for result in suite.category_results:
        s = result.category_score
        if s is not None:
            session.add(
                CategoryScoreRow(
                    run_id=run_id,
                    category_id=result.category_id,
                    score=s.score,
                    num_tests=s.num_tests,
                    num_passed=s.num_passed,
                    num_failed=s.num_failed,
                )
            )
        for ex in result.exchanges:
            v = ex.verdict
            session.add(
                CaseResult(
                    run_id=run_id,
                    category_id=ex.test_case.category_id,
                    test_case_id=ex.test_case.id,
                    probe_type=ex.test_case.probe_type,
                    prompt=ex.test_case.prompt,
                    expected_behavior=ex.test_case.expected_behavior,
                    response_text=ex.response.text,
                    passed=(v.passed if v else None),
                    severity=(v.severity if v else None),
                    reason=(v.reason if v else None),
                )
            )

    for m in report.mitigations:
        session.add(
            MitigationRow(
                run_id=run_id,
                category_id=m.category_id,
                issue=m.issue,
                suggestion=m.suggestion,
                priority=m.priority,
            )
        )

    session.commit()
    session.refresh(run)
    return run


def get_run(session: Session, run_id: str) -> Run | None:
    return session.get(Run, run_id)


def get_run_report(session: Session, run_id: str) -> RobustnessReport | None:
    """Reconstruct the full RobustnessReport for a completed run."""
    run = session.get(Run, run_id)
    if run is None or not run.report_json:
        return None
    return RobustnessReport.model_validate_json(run.report_json)


def list_runs(
    session: Session, *, target_id: str | None = None, limit: int = 50
) -> list[Run]:
    """List runs (optionally filtered by target), newest first."""
    stmt = select(Run).order_by(Run.created_at.desc()).limit(limit)
    if target_id is not None:
        stmt = stmt.where(Run.target_id == target_id)
    return list(session.scalars(stmt))
