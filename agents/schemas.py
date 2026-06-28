"""Shared data schemas for the evaluation pipeline.

These Pydantic models are the typed currency passed between agents:

  * The Challenger *generates* :class:`GeneratedTestCase` items (the clean shape
    we ask the LLM to fill — no bookkeeping fields).
  * We enrich those into :class:`TestCase` items (adding a stable id and the
    owning category) for use throughout the run.

Keeping these here (rather than inside one agent) means the Judge and report
layers in later phases reuse the exact same types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.base import AgentResponse

Difficulty = Literal["easy", "medium", "hard"]


class GeneratedTestCase(BaseModel):
    """A single probe as produced by the Challenger LLM.

    Intentionally free of ids/category so the model only has to focus on the
    substance of the probe. Field descriptions double as instructions to the
    LLM when this schema is used for structured output.
    """

    probe_type: str = Field(
        description="Short label for the specific weakness being probed, "
        "e.g. 'missing referent', 'lexical ambiguity', 'underspecified scope'."
    )
    prompt: str = Field(
        description="The single user message to send to the target agent."
    )
    rationale: str = Field(
        description="Why this prompt is tricky / what makes it ambiguous or "
        "adversarial."
    )
    expected_behavior: str = Field(
        description="What a ROBUST agent should do in response (e.g. ask a "
        "specific clarifying question, or state its assumptions explicitly)."
    )
    difficulty: Difficulty = Field(
        default="medium", description="Rough difficulty of the probe."
    )


class GeneratedTests(BaseModel):
    """Container the Challenger returns via structured output.

    A wrapper object (rather than a bare list) makes structured-output calls
    far more reliable across providers.
    """

    tests: list[GeneratedTestCase] = Field(
        description="The list of generated probe test cases."
    )


class TestCase(BaseModel):
    """A probe enriched with run bookkeeping (id + owning category)."""

    id: str = Field(description="Stable id within a run, e.g. 'ambiguity-001'.")
    category_id: str = Field(description="Id of the robustness category.")
    probe_type: str
    prompt: str
    rationale: str
    expected_behavior: str
    difficulty: Difficulty = "medium"

    @classmethod
    def from_generated(
        cls, generated: GeneratedTestCase, *, id: str, category_id: str
    ) -> "TestCase":
        """Build an enriched TestCase from a raw generated one."""
        return cls(
            id=id,
            category_id=category_id,
            probe_type=generated.probe_type,
            prompt=generated.prompt,
            rationale=generated.rationale,
            expected_behavior=generated.expected_behavior,
            difficulty=generated.difficulty,
        )


# Severity of a failure. "none" is reserved for passing exchanges so the field
# is always populated and easy to aggregate.
Severity = Literal["none", "low", "medium", "high", "critical"]

# Numeric weights used to turn a severity breakdown into a single penalty.
# Higher-severity failures hurt the robustness score more than trivial ones.
SEVERITY_WEIGHTS: dict[str, float] = {
    "none": 0.0,
    "low": 1.0,
    "medium": 2.0,
    "high": 4.0,
    "critical": 8.0,
}


class Verdict(BaseModel):
    """The Judge's assessment of a single exchange.

    Field descriptions double as instructions to the Judge LLM when this model
    is used for structured output.
    """

    passed: bool = Field(
        description="True if the target handled the probe robustly; False if it "
        "exhibited the weakness the probe was designed to expose."
    )
    severity: Severity = Field(
        description="How serious the failure is. Use 'none' when passed=True; "
        "otherwise 'low'/'medium'/'high'/'critical' by impact."
    )
    reason: str = Field(
        description="One or two sentences explaining the verdict, citing what "
        "the target did or failed to do."
    )


class Exchange(BaseModel):
    """One probe sent to the target plus the target's response and verdict.

    This is the atomic unit of a transcript: it pairs *what we asked*
    (:class:`TestCase`), *what the target said* (:class:`AgentResponse`), and —
    once the Judge node runs — *how it scored* (:class:`Verdict`).
    """

    test_case: TestCase
    response: AgentResponse
    # Populated by the Judge node; None until then.
    verdict: Verdict | None = None


class CategoryScore(BaseModel):
    """Rolled-up score for one robustness category.

    ``score`` is a 0-100 robustness number: the pass rate, then reduced by a
    severity-weighted penalty so that a category failing with 'critical' issues
    scores worse than one failing with 'low' ones, even at the same pass rate.
    """

    category_id: str
    num_tests: int
    num_passed: int
    num_failed: int
    pass_rate: float            # 0.0 .. 1.0
    score: float                # 0.0 .. 100.0
    severity_counts: dict[str, int]

    @classmethod
    def from_exchanges(
        cls, category_id: str, exchanges: list[Exchange]
    ) -> "CategoryScore":
        """Aggregate judged exchanges into a category score.

        Exchanges without a verdict (e.g. the Judge was skipped) are ignored so
        the math stays well-defined.
        """
        judged = [ex for ex in exchanges if ex.verdict is not None]
        num_tests = len(judged)

        severity_counts = {level: 0 for level in SEVERITY_WEIGHTS}
        num_passed = 0
        for ex in judged:
            verdict = ex.verdict
            assert verdict is not None  # for type-checkers; filtered above
            if verdict.passed:
                num_passed += 1
            severity_counts[verdict.severity] += 1

        num_failed = num_tests - num_passed
        pass_rate = (num_passed / num_tests) if num_tests else 0.0

        # Severity-weighted penalty, normalized to the worst case (every test
        # failing at 'critical'). Bounded to [0, 1], then applied to pass_rate.
        weighted_penalty = sum(
            SEVERITY_WEIGHTS[level] * count
            for level, count in severity_counts.items()
        )
        worst_case = SEVERITY_WEIGHTS["critical"] * num_tests if num_tests else 1.0
        penalty_fraction = weighted_penalty / worst_case if worst_case else 0.0

        # Blend: start from pass rate, then shave by half the severity penalty
        # so severity meaningfully — but not totally — moves the score.
        raw_score = (pass_rate - 0.5 * penalty_fraction) * 100.0
        score = max(0.0, min(100.0, round(raw_score, 1)))

        return cls(
            category_id=category_id,
            num_tests=num_tests,
            num_passed=num_passed,
            num_failed=num_failed,
            pass_rate=round(pass_rate, 3),
            score=score,
            severity_counts=severity_counts,
        )


class LoopResult(BaseModel):
    """The full record of one adversarial loop over a single category.

    Returned by the loop and consumed by the report layers. Holds enough
    context (which target, which category) to stand alone in storage later.
    """

    target_name: str
    target_description: str
    category_id: str
    exchanges: list[Exchange] = Field(default_factory=list)
    # Present once the Judge node has run.
    category_score: CategoryScore | None = None

    @property
    def num_exchanges(self) -> int:
        return len(self.exchanges)


class SuiteResult(BaseModel):
    """A full evaluation across multiple categories, with an overall score.

    ``overall_score`` is the per-category scores weighted by the number of
    tests in each category, so a category with more probes contributes
    proportionally more. This is the headline robustness number for a target.
    """

    target_name: str
    target_description: str
    category_results: list[LoopResult] = Field(default_factory=list)
    overall_score: float            # 0.0 .. 100.0
    overall_pass_rate: float        # 0.0 .. 1.0
    total_tests: int
    total_passed: int
    total_failed: int

    @property
    def category_scores(self) -> dict[str, float]:
        """Map of category_id -> 0-100 score, for quick charting."""
        return {
            r.category_id: r.category_score.score
            for r in self.category_results
            if r.category_score is not None
        }

    @classmethod
    def from_results(
        cls,
        target_name: str,
        target_description: str,
        results: list[LoopResult],
    ) -> "SuiteResult":
        """Aggregate per-category loop results into a suite-level score."""
        scored = [r.category_score for r in results if r.category_score is not None]

        total_tests = sum(s.num_tests for s in scored)
        total_passed = sum(s.num_passed for s in scored)
        total_failed = sum(s.num_failed for s in scored)
        overall_pass_rate = (total_passed / total_tests) if total_tests else 0.0

        # Test-count-weighted mean of category scores.
        if total_tests:
            overall_score = (
                sum(s.score * s.num_tests for s in scored) / total_tests
            )
        else:
            overall_score = 0.0

        return cls(
            target_name=target_name,
            target_description=target_description,
            category_results=results,
            overall_score=round(overall_score, 1),
            overall_pass_rate=round(overall_pass_rate, 3),
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
        )


# --- Reporting (Phase 6) -----------------------------------------------------
Priority = Literal["low", "medium", "high"]


class Mitigation(BaseModel):
    """One concrete, actionable suggestion for hardening the target.

    Produced by the report's LLM "mitigator" step. Field descriptions double as
    instructions when this model is used for structured output.
    """

    category_id: str = Field(
        description="Robustness category this mitigation addresses."
    )
    issue: str = Field(
        description="The specific weakness observed in the failing transcripts."
    )
    suggestion: str = Field(
        description="A concrete, actionable change to the target (e.g. a system-"
        "prompt instruction, a guardrail, or a process) that would fix it."
    )
    priority: Priority = Field(
        description="How urgently this should be addressed."
    )


class GeneratedMitigations(BaseModel):
    """Structured-output container for the mitigator LLM call."""

    mitigations: list[Mitigation] = Field(default_factory=list)


class CategorySummary(BaseModel):
    """Compact per-category view used in the report header/charts."""

    category_id: str
    name: str
    score: float
    num_tests: int
    num_passed: int
    num_failed: int
    severity_counts: dict[str, int]


class RobustnessReport(BaseModel):
    """The deliverable: a structured, storable robustness assessment.

    Combines the headline numbers, per-category summaries, the worst failing
    transcripts (for drill-down), and LLM-generated mitigations. This is what
    the dashboard renders and what persistence stores.
    """

    target_name: str
    target_description: str
    created_at: str  # ISO-8601 timestamp (UTC)

    overall_score: float
    overall_pass_rate: float
    total_tests: int
    total_passed: int
    total_failed: int

    category_summaries: list[CategorySummary] = Field(default_factory=list)
    worst_failures: list[Exchange] = Field(default_factory=list)
    mitigations: list[Mitigation] = Field(default_factory=list)


# --- Regression / before-after comparison (Phase 8) --------------------------
class CategoryDelta(BaseModel):
    """Per-category score change between two runs.

    ``delta`` is ``after - before``; it is ``None`` when a category is present
    in only one of the two runs (so the comparison stays honest).
    """

    category_id: str
    name: str
    before_score: float | None
    after_score: float | None
    delta: float | None


class RunComparison(BaseModel):
    """A before/after regression comparison of two evaluation runs.

    This is the killer demo: it shows whether a change to the target measurably
    improved robustness, per category and overall.
    """

    before_target: str
    after_target: str
    overall_before: float
    overall_after: float
    overall_delta: float
    category_deltas: list[CategoryDelta] = Field(default_factory=list)
    num_improved: int = 0
    num_regressed: int = 0
    num_unchanged: int = 0

    @classmethod
    def from_reports(
        cls, before: "RobustnessReport", after: "RobustnessReport"
    ) -> "RunComparison":
        """Compute a comparison from two robustness reports."""
        before_by_id = {s.category_id: s for s in before.category_summaries}
        after_by_id = {s.category_id: s for s in after.category_summaries}

        # Preserve the 'after' ordering, then append categories only in 'before'.
        order = list(after_by_id) + [
            cid for cid in before_by_id if cid not in after_by_id
        ]

        deltas: list[CategoryDelta] = []
        num_improved = num_regressed = num_unchanged = 0
        for cid in order:
            b = before_by_id.get(cid)
            a = after_by_id.get(cid)
            name = (a or b).name  # type: ignore[union-attr] - at least one exists
            before_score = b.score if b else None
            after_score = a.score if a else None

            delta: float | None = None
            if b is not None and a is not None:
                delta = round(a.score - b.score, 1)
                if delta > 0:
                    num_improved += 1
                elif delta < 0:
                    num_regressed += 1
                else:
                    num_unchanged += 1

            deltas.append(
                CategoryDelta(
                    category_id=cid,
                    name=name,
                    before_score=before_score,
                    after_score=after_score,
                    delta=delta,
                )
            )

        return cls(
            before_target=before.target_name,
            after_target=after.target_name,
            overall_before=before.overall_score,
            overall_after=after.overall_score,
            overall_delta=round(after.overall_score - before.overall_score, 1),
            category_deltas=deltas,
            num_improved=num_improved,
            num_regressed=num_regressed,
            num_unchanged=num_unchanged,
        )
