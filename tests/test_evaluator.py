"""Tests for the multi-category suite evaluator (Phase 5).

Fully offline: a fake challenger factory + fake judge + EchoTaskAgent let the
real evaluator/loop run across several categories with no network. We also
check the category catalog grew and the suite-level aggregation math.
"""

from __future__ import annotations

from agents.base import EchoTaskAgent
from agents.categories import (
    AMBIGUITY,
    CATEGORIES,
    FACTUAL_CONSISTENCY,
    LOGICAL_TRAPS,
    RobustnessCategory,
    list_categories,
)
from agents.evaluator import RedTeamEvaluator
from agents.schemas import (
    CategoryScore,
    Exchange,
    LoopResult,
    SuiteResult,
    TestCase,
    Verdict,
)


# --- fakes -------------------------------------------------------------------
class _FakeChallenger:
    def __init__(self, category: RobustnessCategory, n: int = 2) -> None:
        self.category = category
        self._n = n

    def generate(self, target_description: str, n: int) -> list[TestCase]:
        return [
            TestCase(
                id=f"{self.category.id}-{i:03d}",
                category_id=self.category.id,
                probe_type="t",
                prompt=f"probe {i}",
                rationale="r",
                expected_behavior="e",
            )
            for i in range(1, n + 1)
        ]


class _FakeJudge:
    """Passes the first probe of each category, fails the rest at 'high'."""

    def evaluate(self, exchange: Exchange, category) -> Verdict:  # noqa: ANN001
        first = exchange.test_case.id.endswith("-001")
        if first:
            return Verdict(passed=True, severity="none", reason="ok")
        return Verdict(passed=False, severity="high", reason="bad")


def _factory(category: RobustnessCategory):
    return _FakeChallenger(category)


# --- catalog -----------------------------------------------------------------
def test_six_categories_registered() -> None:
    ids = set(CATEGORIES)
    assert {
        "ambiguity",
        "factual_consistency",
        "uncertainty_calibration",
        "instruction_following",
        "logical_traps",
        "self_contradiction",
    } <= ids
    assert len(list_categories()) >= 6


# --- suite run ---------------------------------------------------------------
def test_suite_runs_selected_categories_and_aggregates() -> None:
    evaluator = RedTeamEvaluator(
        target=EchoTaskAgent(),
        judge=_FakeJudge(),
        challenger_factory=_factory,
        categories=[AMBIGUITY, FACTUAL_CONSISTENCY, LOGICAL_TRAPS],
    )

    suite = evaluator.run(tests_per_category=2)

    assert isinstance(suite, SuiteResult)
    # One LoopResult per selected category.
    assert len(suite.category_results) == 3
    assert set(suite.category_scores) == {
        "ambiguity",
        "factual_consistency",
        "logical_traps",
    }
    # 3 categories x 2 probes; 1 pass + 1 fail each => 3 passed, 3 failed.
    assert suite.total_tests == 6
    assert suite.total_passed == 3
    assert suite.total_failed == 3
    assert suite.overall_pass_rate == 0.5
    assert 0.0 <= suite.overall_score <= 100.0


def test_suite_defaults_to_full_catalog() -> None:
    evaluator = RedTeamEvaluator(
        target=EchoTaskAgent(),
        judge=_FakeJudge(),
        challenger_factory=_factory,
    )
    assert len(evaluator.categories) == len(list_categories())


def test_overall_score_is_test_count_weighted() -> None:
    """A category with more tests should pull the overall score harder."""
    # Category A: 1 test, perfect (100). Category B: 3 tests, all fail (low).
    a = LoopResult(
        target_name="t",
        target_description="d",
        category_id="ambiguity",
        category_score=CategoryScore(
            category_id="ambiguity",
            num_tests=1,
            num_passed=1,
            num_failed=0,
            pass_rate=1.0,
            score=100.0,
            severity_counts={"none": 1, "low": 0, "medium": 0, "high": 0, "critical": 0},
        ),
    )
    b = LoopResult(
        target_name="t",
        target_description="d",
        category_id="logical_traps",
        category_score=CategoryScore(
            category_id="logical_traps",
            num_tests=3,
            num_passed=0,
            num_failed=3,
            pass_rate=0.0,
            score=0.0,
            severity_counts={"none": 0, "low": 0, "medium": 0, "high": 3, "critical": 0},
        ),
    )
    suite = SuiteResult.from_results("t", "d", [a, b])
    # Weighted: (100*1 + 0*3) / 4 = 25, NOT the simple mean of 50.
    assert suite.overall_score == 25.0
