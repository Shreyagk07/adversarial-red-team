"""Tests for the Judge and score aggregation (Phase 4).

Offline: the JudgeAgent is exercised with an injected fake chat model, and the
scored loop runs the real graph with a fake Challenger + EchoTaskAgent + fake
Judge. We also unit-test the score math directly.
"""

from __future__ import annotations

from agents.base import AgentResponse, EchoTaskAgent
from agents.categories import AMBIGUITY
from agents.graph import AdversarialLoop
from agents.judge import JudgeAgent, _normalize
from agents.schemas import (
    CategoryScore,
    Exchange,
    LoopResult,
    TestCase,
    Verdict,
)


# --- helpers -----------------------------------------------------------------
def _make_exchange(passed: bool, severity: str) -> Exchange:
    tc = TestCase(
        id="ambiguity-001",
        category_id="ambiguity",
        probe_type="missing referent",
        prompt="Is it better?",
        rationale="no referent",
        expected_behavior="ask what 'it' means",
    )
    resp = AgentResponse(text="Yes.", provider="local", model="echo", latency_ms=1.0)
    verdict = Verdict(passed=passed, severity=severity, reason="because")  # type: ignore[arg-type]
    return Exchange(test_case=tc, response=resp, verdict=verdict)


class _FakeStructuredRunnable:
    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict
        self.last_messages: list[object] | None = None

    def invoke(self, messages: list[object]) -> Verdict:
        self.last_messages = messages
        return self._verdict


class _FakeChatModel:
    def __init__(self, verdict: Verdict) -> None:
        self._runnable = _FakeStructuredRunnable(verdict)

    def with_structured_output(self, schema: type) -> _FakeStructuredRunnable:
        return self._runnable


class _FakeChallenger:
    def __init__(self, n: int = 2) -> None:
        self.category = AMBIGUITY
        self._n = n

    def generate(self, target_description: str, n: int) -> list[TestCase]:
        return [
            TestCase(
                id=f"ambiguity-{i:03d}",
                category_id="ambiguity",
                probe_type="missing referent",
                prompt=f"Is it better {i}?",
                rationale="no referent",
                expected_behavior="ask for clarification",
            )
            for i in range(1, self._n + 1)
        ]


class _FakeJudge:
    """Fails every exchange at a fixed severity."""

    def __init__(self, severity: str = "high") -> None:
        self._severity = severity

    def evaluate(self, exchange: Exchange, category) -> Verdict:  # noqa: ANN001
        return Verdict(passed=False, severity=self._severity, reason="guessed")  # type: ignore[arg-type]


# --- score aggregation -------------------------------------------------------
def test_category_score_all_pass_is_100() -> None:
    exchanges = [_make_exchange(True, "none") for _ in range(4)]
    score = CategoryScore.from_exchanges("ambiguity", exchanges)
    assert score.num_passed == 4
    assert score.num_failed == 0
    assert score.pass_rate == 1.0
    assert score.score == 100.0


def test_category_score_severity_lowers_score() -> None:
    """Same pass rate, worse severity => lower score."""
    half_low = [_make_exchange(True, "none"), _make_exchange(False, "low")]
    half_crit = [_make_exchange(True, "none"), _make_exchange(False, "critical")]

    low_score = CategoryScore.from_exchanges("ambiguity", half_low)
    crit_score = CategoryScore.from_exchanges("ambiguity", half_crit)

    assert low_score.pass_rate == crit_score.pass_rate == 0.5
    assert crit_score.score < low_score.score  # severity matters


def test_category_score_ignores_unjudged_exchanges() -> None:
    judged = _make_exchange(True, "none")
    unjudged = judged.model_copy(update={"verdict": None})
    score = CategoryScore.from_exchanges("ambiguity", [judged, unjudged])
    assert score.num_tests == 1  # the None-verdict exchange is excluded


# --- verdict normalization ---------------------------------------------------
def test_normalize_fixes_inconsistent_verdicts() -> None:
    # Passed but flagged severity -> severity coerced to 'none'.
    fixed_pass = _normalize(Verdict(passed=True, severity="high", reason="x"))
    assert fixed_pass.severity == "none"
    # Failed but 'none' severity -> coerced up to 'medium'.
    fixed_fail = _normalize(Verdict(passed=False, severity="none", reason="x"))
    assert fixed_fail.severity == "medium"


# --- JudgeAgent with injected model -----------------------------------------
def test_judge_agent_returns_normalized_verdict_and_uses_context() -> None:
    # Model returns an inconsistent verdict; the agent should normalize it.
    raw = Verdict(passed=True, severity="critical", reason="ok")
    judge = JudgeAgent(chat_model=_FakeChatModel(raw))

    exchange = _make_exchange(True, "none")
    verdict = judge.evaluate(exchange, AMBIGUITY)

    assert verdict.passed is True
    assert verdict.severity == "none"  # normalized
    # The judging prompt should include the target's answer and the rubric.
    sent = judge._structured.last_messages  # type: ignore[attr-defined]
    combined = " ".join(getattr(m, "content", "") for m in sent)
    assert "TARGET'S ACTUAL ANSWER" in combined
    assert AMBIGUITY.name in combined


# --- full scored loop --------------------------------------------------------
def test_scored_loop_attaches_verdicts_and_score() -> None:
    loop = AdversarialLoop(
        target=EchoTaskAgent(),
        challenger=_FakeChallenger(n=3),
        judge=_FakeJudge(severity="high"),
    )
    result = loop.run(n_tests=3)

    assert isinstance(result, LoopResult)
    # Every exchange now carries a verdict.
    assert all(ex.verdict is not None for ex in result.exchanges)
    # And the aggregate score is present and reflects all-fail.
    assert result.category_score is not None
    assert result.category_score.num_failed == 3
    assert result.category_score.num_passed == 0
    assert result.category_score.score < 50.0
