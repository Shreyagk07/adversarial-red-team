"""HTTP API routes for targets, runs, and reports.

Endpoints:
  POST   /targets                 register a target
  GET    /targets                 list targets
  GET    /targets/{id}            fetch a target
  POST   /targets/{id}/evaluate   launch an evaluation (returns 202 + run id)
  GET    /runs                    list runs (optionally ?target_id=)
  GET    /runs/{id}               run status + headline metrics
  GET    /runs/{id}/report        full robustness report (when completed)

The evaluate endpoint launches the work as a FastAPI BackgroundTask and returns
immediately — evaluations make many LLM calls and would otherwise time out the
request. The client polls GET /runs/{id} until status is 'completed'.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from agents.categories import list_categories
from agents.schemas import RobustnessReport, RunComparison
from backend.api_models import (
    CategoryOut,
    EvaluateRequest,
    RunOut,
    TargetCreate,
    TargetOut,
)
from backend.config import get_settings
from backend.service import run_evaluation_job
from storage import repository as repo
from storage.db import get_db

router = APIRouter()


# --- Categories --------------------------------------------------------------
@router.get("/categories", response_model=list[CategoryOut], tags=["meta"])
def get_categories() -> list[CategoryOut]:
    """List the configured robustness categories."""
    return [
        CategoryOut(id=c.id, name=c.name, description=c.description)
        for c in list_categories()
    ]


# --- Targets -----------------------------------------------------------------
@router.post("/targets", response_model=TargetOut, status_code=status.HTTP_201_CREATED, tags=["targets"])
def register_target(body: TargetCreate, db: Session = Depends(get_db)) -> TargetOut:
    target = repo.create_target(
        db,
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        provider=body.provider,
        model=body.model,
        temperature=body.temperature,
    )
    return TargetOut.model_validate(target)


@router.get("/targets", response_model=list[TargetOut], tags=["targets"])
def get_targets(db: Session = Depends(get_db)) -> list[TargetOut]:
    return [TargetOut.model_validate(t) for t in repo.list_targets(db)]


@router.get("/targets/{target_id}", response_model=TargetOut, tags=["targets"])
def get_single_target(target_id: str, db: Session = Depends(get_db)) -> TargetOut:
    target = repo.get_target(db, target_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target not found")
    return TargetOut.model_validate(target)


# --- Runs --------------------------------------------------------------------
@router.post(
    "/targets/{target_id}/evaluate",
    response_model=RunOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["runs"],
)
def launch_evaluation(
    target_id: str,
    body: EvaluateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> RunOut:
    target = repo.get_target(db, target_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target not found")

    # Fail fast (and clearly) if the provider key for this target is missing,
    # rather than letting the background job fail opaquely later.
    settings = get_settings()
    key = settings.groq_api_key if target.provider == "groq" else settings.gemini_api_key
    if not key:
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured for provider '{target.provider}'. "
            "Set the corresponding key in the server environment.",
        )

    run = repo.create_run(
        db, target_id=target_id, tests_per_category=body.tests_per_category
    )
    background_tasks.add_task(
        run_evaluation_job,
        run.id,
        target_id,
        body.tests_per_category,
        body.category_ids,
    )
    return RunOut.model_validate(run)


@router.get("/runs", response_model=list[RunOut], tags=["runs"])
def get_runs(
    target_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[RunOut]:
    return [RunOut.model_validate(r) for r in repo.list_runs(db, target_id=target_id)]


@router.get("/runs/{run_id}", response_model=RunOut, tags=["runs"])
def get_single_run(run_id: str, db: Session = Depends(get_db)) -> RunOut:
    run = repo.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunOut.model_validate(run)


@router.get("/runs/{run_id}/report", response_model=RobustnessReport, tags=["runs"])
def get_run_report(run_id: str, db: Session = Depends(get_db)) -> RobustnessReport:
    run = repo.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    report = repo.get_run_report(db, run_id)
    if report is None:
        raise HTTPException(
            status_code=409,
            detail=f"Report not available; run status is '{run.status}'.",
        )
    return report


@router.get("/compare", response_model=RunComparison, tags=["runs"])
def compare_runs(
    before: str = Query(..., description="Run id of the baseline (before) run."),
    after: str = Query(..., description="Run id of the improved (after) run."),
    db: Session = Depends(get_db),
) -> RunComparison:
    """Compute a before/after regression comparison between two completed runs."""
    before_report = repo.get_run_report(db, before)
    after_report = repo.get_run_report(db, after)
    if before_report is None:
        raise HTTPException(
            status_code=409, detail=f"No completed report for 'before' run {before!r}."
        )
    if after_report is None:
        raise HTTPException(
            status_code=409, detail=f"No completed report for 'after' run {after!r}."
        )
    return RunComparison.from_reports(before_report, after_report)
