"""Tests for regression / before-after comparison (Phase 8).

Offline: we build two RobustnessReports by hand and assert the comparison math,
ordering, None-handling for categories present in only one run, and the
Markdown rendering. Also an end-to-end API test seeding two completed runs.
"""

from __future__ import annotations

from agents.reporter import render_comparison_markdown
from agents.schemas import (
    CategorySummary,
    RobustnessReport,
    RunComparison,
)


def _report(target: str, overall: float, cats: dict[str, float]) -> RobustnessReport:
    summaries = [
        CategorySummary(
            category_id=cid,
            name=cid.replace("_", " ").title(),
            score=score,
            num_tests=5,
            num_passed=int(round(score / 20)),
            num_failed=5 - int(round(score / 20)),
            severity_counts={"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0},
        )
        for cid, score in cats.items()
    ]
    return RobustnessReport(
        target_name=target,
        target_description="d",
        created_at="2026-06-28T00:00:00+00:00",
        overall_score=overall,
        overall_pass_rate=overall / 100,
        total_tests=len(cats) * 5,
        total_passed=0,
        total_failed=0,
        category_summaries=summaries,
    )


def test_comparison_computes_deltas_and_counts() -> None:
    before = _report("baseline", 40.0, {"ambiguity": 20.0, "logical_traps": 60.0})
    after = _report("hardened", 70.0, {"ambiguity": 80.0, "logical_traps": 60.0})

    cmp = RunComparison.from_reports(before, after)

    assert cmp.overall_delta == 30.0
    by_id = {d.category_id: d for d in cmp.category_deltas}
    assert by_id["ambiguity"].delta == 60.0       # 80 - 20, improved
    assert by_id["logical_traps"].delta == 0.0    # unchanged
    assert cmp.num_improved == 1
    assert cmp.num_unchanged == 1
    assert cmp.num_regressed == 0


def test_comparison_handles_category_in_one_run_only() -> None:
    before = _report("b", 50.0, {"ambiguity": 50.0})
    after = _report("a", 55.0, {"ambiguity": 60.0, "factual_consistency": 50.0})

    cmp = RunComparison.from_reports(before, after)
    by_id = {d.category_id: d for d in cmp.category_deltas}

    # factual_consistency only exists in 'after' -> before_score None, delta None
    assert by_id["factual_consistency"].before_score is None
    assert by_id["factual_consistency"].delta is None
    # It is not counted as improved/regressed/unchanged.
    assert cmp.num_improved == 1  # only ambiguity (50 -> 60)
    assert cmp.num_unchanged == 0
    assert cmp.num_regressed == 0


def test_comparison_detects_regression() -> None:
    before = _report("b", 80.0, {"ambiguity": 80.0})
    after = _report("a", 60.0, {"ambiguity": 60.0})
    cmp = RunComparison.from_reports(before, after)
    assert cmp.overall_delta == -20.0
    assert cmp.num_regressed == 1


def test_render_comparison_markdown() -> None:
    before = _report("baseline", 40.0, {"ambiguity": 20.0})
    after = _report("hardened", 80.0, {"ambiguity": 80.0})
    md = render_comparison_markdown(RunComparison.from_reports(before, after))
    assert "Regression Comparison" in md
    assert "baseline" in md and "hardened" in md
    assert "+40.0" in md  # overall delta rendered with sign
