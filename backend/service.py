"""Evaluation service — wires a persisted Target to the agent pipeline.

This is the bridge between the API/storage world (Target rows, Run rows) and
the agents world (Task agent, evaluator, reporter). It is designed to run as a
background job so the HTTP request can return immediately with a run id while
the (multi-call, slow) evaluation proceeds.
"""

from __future__ import annotations

from agents.categories import RobustnessCategory, get_category
from agents.evaluator import RedTeamEvaluator
from agents.observability import observe_run
from agents.reporter import Reporter
from agents.task_agent import DEFAULT_SYSTEM_PROMPT, LLMTaskAgent
from backend.config import get_settings
from storage import repository as repo
from storage.db import session_scope
from storage.models import Target


def build_target_agent(target: Target) -> LLMTaskAgent:
    """Construct an LLM Task agent from a persisted Target configuration."""
    settings = get_settings()
    return LLMTaskAgent(
        settings,
        name=target.name,
        description=target.description or target.name,
        system_prompt=target.system_prompt or DEFAULT_SYSTEM_PROMPT,
        provider=target.provider,  # type: ignore[arg-type]
        model=target.model,
        temperature=target.temperature,
    )


def run_evaluation_job(
    run_id: str,
    target_id: str,
    tests_per_category: int,
    category_ids: list[str] | None,
) -> None:
    """Execute a full evaluation and persist results. Safe for background use.

    Any failure is caught and recorded on the run (status='failed') so a crash
    never leaves a run stuck in 'running' or takes down the worker.
    """
    try:
        with session_scope() as session:
            target = repo.get_target(session, target_id)
            if target is None:
                repo.mark_run_failed(session, run_id, "Target not found")
                return

            agent = build_target_agent(target)

            categories: list[RobustnessCategory] | None = None
            if category_ids:
                categories = [get_category(cid) for cid in category_ids]

            evaluator = RedTeamEvaluator.from_settings(
                agent, get_settings(), categories=categories
            )
            # Group all of this run's agent calls under one Langfuse trace
            # (no-op when Langfuse isn't configured).
            with observe_run(f"evaluation:{run_id}"):
                suite = evaluator.run(tests_per_category=tests_per_category)
                report = Reporter(get_settings()).generate(suite)

            repo.complete_run(session, run_id, suite=suite, report=report)
    except Exception as exc:  # noqa: BLE001 - we persist any failure
        # New scope: the failing one may have rolled back.
        with session_scope() as session:
            repo.mark_run_failed(session, run_id, f"{type(exc).__name__}: {exc}")
