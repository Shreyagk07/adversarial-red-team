"""Tests for report assembly, rendering, and persistence (Phase 6).

Offline: we build a SuiteResult by hand (varied severities), then exercise the
deterministic parts (worst-failure sorting, summaries), the LLM mitigation path
via an injected fake model, the graceful no-LLM degradation, Markdown
rendering, and saving to disk.
"""

from __future__ import annotations

from pathlib import Path

from agents.base import AgentResponse
from agents.reporter import Reporter, render_markdown, save_report
from agents.schemas import (
    CategoryScore,
    Exchange,
    GeneratedMitigations,
    LoopResult,
    Mitigation,
    SuiteResult,
    TestCase,
    Verdict,
)
from backend.config import Settings


# --- builders ----------------------------------------------------------------
def _exchange(cat: str, idx: int, *, passed: bool, severity: str) -> Exchange:
    tc = TestCase(
        id=f"{cat}-{idx:03d}",
        category_id=cat,
        probe_type="t",
        prompt=f"{cat} probe {idx}",
        rationale="r",
        expected_behavior="be robust",
    )
    resp = AgentResponse(text="some answer", provider="local", model="echo", latency_ms=1.0)
    verdict = Verdict(passed=passed, severity=severity, reason=f"{severity} issue")  # type: ignore[arg-type]
    return Exchange(test_case=tc, response=resp, verdict=verdict)


def _loop_result(cat: str, exchanges: list[Exchange]) -> LoopResult:
    return LoopResult(
        target_name="demo",
        target_description="a demo target",
        category_id=cat,
        exchanges=exchanges,
        category_score=CategoryScore.from_exchanges(cat, exchanges),
    )


def _sample_suite() -> SuiteResult:
    amb = _loop_result(
        "ambiguity",
        [
            _exchange("ambiguity", 1, passed=True, severity="none"),
            _exchange("ambiguity", 2, passed=False, severity="low"),
        ],
    )
    logic = _loop_result(
        "logical_traps",
        [
            _exchange("logical_traps", 1, passed=False, severity="critical"),
            _exchange("logical_traps", 2, passed=False, severity="medium"),
        ],
    )
    return SuiteResult.from_results("demo", "a demo target", [amb, logic])


# --- fakes for the mitigator -------------------------------------------------
class _FakeStructured:
    def __init__(self, result: GeneratedMitigations) -> None:
        self._result = result

    def invoke(self, messages: list[object]) -> GeneratedMitigations:
        return self._result


class _FakeChatModel:
    def __init__(self, result: GeneratedMitigations) -> None:
        self._r = result

    def with_structured_output(self, schema: type) -> _FakeStructured:
        return _FakeStructured(self._r)


# --- deterministic assembly --------------------------------------------------
def test_worst_failures_sorted_by_severity_and_capped() -> None:
    suite = _sample_suite()
    # No key + no injected model => mitigations disabled, but assembly works.
    reporter = Reporter(Settings(groq_api_key=None), max_worst=2)
    report = reporter.generate(suite, now="2026-06-28T00:00:00+00:00")

    assert reporter.mitigations_enabled is False
    assert report.mitigations == []
    # Three failures exist (low, critical, medium); capped at 2, most severe first.
    assert len(report.worst_failures) == 2
    severities = [ex.verdict.severity for ex in report.worst_failures]  # type: ignore[union-attr]
    assert severities == ["critical", "medium"]


def test_category_summaries_built_with_names() -> None:
    suite = _sample_suite()
    reporter = Reporter(Settings(groq_api_key=None))
    report = reporter.generate(suite, now="2026-06-28T00:00:00+00:00")

    by_id = {s.category_id: s for s in report.category_summaries}
    assert by_id["ambiguity"].name == "Ambiguity handling"
    assert by_id["ambiguity"].num_passed == 1
    assert by_id["logical_traps"].num_failed == 2


# --- mitigation path ---------------------------------------------------------
def test_mitigations_generated_with_injected_model() -> None:
    suite = _sample_suite()
    canned = GeneratedMitigations(
        mitigations=[
            Mitigation(
                category_id="logical_traps",
                issue="falls for cognitive-reflection traps",
                suggestion="add a 'think step by step and re-check' instruction",
                priority="high",
            )
        ]
    )
    reporter = Reporter(Settings(groq_api_key=None), chat_model=_FakeChatModel(canned))
    assert reporter.mitigations_enabled is True

    report = reporter.generate(suite, now="2026-06-28T00:00:00+00:00")
    assert len(report.mitigations) == 1
    assert report.mitigations[0].priority == "high"
    assert report.mitigations[0].category_id == "logical_traps"


# --- rendering ---------------------------------------------------------------
def test_render_markdown_contains_key_sections() -> None:
    suite = _sample_suite()
    canned = GeneratedMitigations(
        mitigations=[
            Mitigation(
                category_id="ambiguity",
                issue="guesses instead of clarifying",
                suggestion="ask a clarifying question when the request is vague",
                priority="medium",
            )
        ]
    )
    reporter = Reporter(Settings(groq_api_key=None), chat_model=_FakeChatModel(canned))
    report = reporter.generate(suite, now="2026-06-28T00:00:00+00:00")

    md = render_markdown(report)
    assert "# Robustness Report" in md
    assert "Overall score" in md
    assert "Ambiguity handling" in md
    assert "Recommended mitigations" in md
    assert "ask a clarifying question" in md


# --- persistence -------------------------------------------------------------
def test_save_report_writes_json_and_markdown(tmp_path: Path) -> None:
    suite = _sample_suite()
    reporter = Reporter(Settings(groq_api_key=None))
    report = reporter.generate(suite, now="2026-06-28T12:34:56+00:00")

    json_path, md_path = save_report(report, base_dir=tmp_path)
    assert json_path.exists() and json_path.suffix == ".json"
    assert md_path.exists() and md_path.suffix == ".md"
    # JSON round-trips back into a report.
    from agents.schemas import RobustnessReport

    reloaded = RobustnessReport.model_validate_json(json_path.read_text("utf-8"))
    assert reloaded.target_name == "demo"
    assert reloaded.overall_score == suite.overall_score
