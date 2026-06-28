"""Report generation — turn a SuiteResult into a robustness report.

This module does three things:

  1. *Deterministic assembly* — compute per-category summaries and surface the
     worst failing transcripts (sorted by severity). No LLM needed; this part
     always runs and is fully unit-tested.
  2. *Mitigations* — one LLM call that reads a compact summary of the failures
     and proposes prioritized, actionable hardening suggestions. Optional: if
     no API key is configured, the report is still produced, just without
     mitigations.
  3. *Rendering + persistence* — render the report to Markdown and save both
     JSON and Markdown under ``storage/reports/`` (the seam Phase 7 upgrades to
     a database).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.categories import get_category
from agents.llm import MissingAPIKeyError, build_chat_model
from agents.schemas import (
    SEVERITY_WEIGHTS,
    CategorySummary,
    Exchange,
    GeneratedMitigations,
    Mitigation,
    RobustnessReport,
    RunComparison,
    SuiteResult,
)
from backend.config import Settings, get_settings

# How many failing transcripts to surface in the report / feed to the mitigator.
DEFAULT_MAX_WORST = 6
# Cap on failures summarized for the LLM, to keep the prompt small.
MAX_FAILURES_FOR_LLM = 12

# Default base directory for saved reports.
DEFAULT_REPORTS_DIR = Path("storage") / "reports"


class Reporter:
    """Builds :class:`RobustnessReport` objects from suite results."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chat_model: BaseChatModel | None = None,
        max_worst: int = DEFAULT_MAX_WORST,
    ) -> None:
        """Create a Reporter.

        Args:
            settings: App settings (provider + keys).
            chat_model: Inject a model (tests). If omitted we try to build one;
                if no key is configured, mitigations are disabled gracefully.
            max_worst: How many worst-failing transcripts to surface.
        """
        self.settings = settings or get_settings()
        self.max_worst = max_worst

        # Mitigations require an LLM. We degrade gracefully if there's no key.
        self._structured = None
        model = chat_model
        if model is None:
            try:
                model = build_chat_model(self.settings)
            except MissingAPIKeyError:
                model = None
        if model is not None:
            self._structured = model.with_structured_output(GeneratedMitigations)

    @property
    def mitigations_enabled(self) -> bool:
        """True when an LLM is available to generate mitigations."""
        return self._structured is not None

    def generate(
        self, suite: SuiteResult, *, now: str | None = None
    ) -> RobustnessReport:
        """Build a full robustness report from a suite result."""
        summaries = [_summarize_category(r) for r in suite.category_results
                     if r.category_score is not None]
        worst = _worst_failures(suite, limit=self.max_worst)
        mitigations = self._mitigate(suite, worst) if self.mitigations_enabled else []

        created_at = now or datetime.now(timezone.utc).isoformat()

        return RobustnessReport(
            target_name=suite.target_name,
            target_description=suite.target_description,
            created_at=created_at,
            overall_score=suite.overall_score,
            overall_pass_rate=suite.overall_pass_rate,
            total_tests=suite.total_tests,
            total_passed=suite.total_passed,
            total_failed=suite.total_failed,
            category_summaries=summaries,
            worst_failures=worst,
            mitigations=mitigations,
        )

    # --- mitigations (LLM) --------------------------------------------------
    def _mitigate(
        self, suite: SuiteResult, worst: list[Exchange]
    ) -> list[Mitigation]:
        """Ask the LLM for prioritized hardening suggestions."""
        assert self._structured is not None
        system = (
            "You are an AI robustness consultant. Given a target agent's "
            "evaluation results and its worst failing transcripts, propose "
            "concrete, actionable mitigations to harden the target. Prefer "
            "specific, implementable changes (a system-prompt instruction, a "
            "guardrail, a verification step) over vague advice. Tie each "
            "mitigation to a category. Keep them defensive — improving the "
            "target's robustness, never enabling misuse."
        )
        user = _summarize_for_llm(suite, worst)
        result: GeneratedMitigations = self._structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return result.mitigations


# --- deterministic helpers (no LLM) -----------------------------------------
def _summarize_category(result) -> CategorySummary:  # noqa: ANN001 - LoopResult
    """Build a CategorySummary from a scored LoopResult."""
    score = result.category_score
    try:
        name = get_category(result.category_id).name
    except KeyError:
        name = result.category_id
    return CategorySummary(
        category_id=result.category_id,
        name=name,
        score=score.score,
        num_tests=score.num_tests,
        num_passed=score.num_passed,
        num_failed=score.num_failed,
        severity_counts=score.severity_counts,
    )


def _worst_failures(suite: SuiteResult, *, limit: int) -> list[Exchange]:
    """Return the most severe failing exchanges across all categories."""
    failures = [
        ex
        for result in suite.category_results
        for ex in result.exchanges
        if ex.verdict is not None and not ex.verdict.passed
    ]
    # Sort by severity weight (descending); stable so input order breaks ties.
    failures.sort(
        key=lambda ex: SEVERITY_WEIGHTS.get(ex.verdict.severity, 0.0),  # type: ignore[union-attr]
        reverse=True,
    )
    return failures[:limit]


def _summarize_for_llm(suite: SuiteResult, worst: list[Exchange]) -> str:
    """Compact text summary of scores + failures to feed the mitigator."""
    lines: list[str] = [
        f"Target: {suite.target_name} — {suite.target_description}",
        f"Overall score: {suite.overall_score}/100 "
        f"(pass rate {suite.overall_pass_rate:.0%}).",
        "",
        "Per-category scores:",
    ]
    for result in suite.category_results:
        s = result.category_score
        if s is None:
            continue
        lines.append(
            f"  - {result.category_id}: {s.score}/100 "
            f"({s.num_passed}/{s.num_tests} passed)"
        )

    lines.append("")
    lines.append("Worst failing transcripts:")
    for ex in worst[:MAX_FAILURES_FOR_LLM]:
        v = ex.verdict
        lines.append(
            f"  [{ex.test_case.category_id} | severity={v.severity if v else '?'}]"
        )
        lines.append(f"    Probe: {ex.test_case.prompt}")
        lines.append(f"    Target answered: {ex.response.text}")
        lines.append(f"    Judge: {v.reason if v else ''}")

    lines.append("")
    lines.append(
        "Propose prioritized mitigations (high/medium/low) tied to categories."
    )
    return "\n".join(lines)


# --- rendering + persistence -------------------------------------------------
def render_markdown(report: RobustnessReport) -> str:
    """Render a report as human-readable Markdown."""
    lines: list[str] = [
        f"# Robustness Report — {report.target_name}",
        "",
        f"_{report.target_description}_",
        "",
        f"**Generated:** {report.created_at}",
        "",
        f"## Overall score: {report.overall_score}/100",
        "",
        f"- Pass rate: **{report.overall_pass_rate:.0%}** "
        f"({report.total_passed}/{report.total_tests} probes passed)",
        "",
        "## Per-category scores",
        "",
        "| Category | Score | Passed | Failed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in report.category_summaries:
        lines.append(
            f"| {s.name} | {s.score:.1f} | {s.num_passed}/{s.num_tests} "
            f"| {s.num_failed} |"
        )

    lines += ["", "## Worst failing transcripts", ""]
    if not report.worst_failures:
        lines.append("_No failures recorded._")
    for i, ex in enumerate(report.worst_failures, start=1):
        v = ex.verdict
        lines += [
            f"### {i}. {ex.test_case.category_id} "
            f"(severity: {v.severity if v else '?'})",
            "",
            f"- **Probe:** {ex.test_case.prompt}",
            f"- **Target answered:** {ex.response.text}",
            f"- **Expected:** {ex.test_case.expected_behavior}",
            f"- **Judge:** {v.reason if v else ''}",
            "",
        ]

    lines += ["## Recommended mitigations", ""]
    if not report.mitigations:
        lines.append("_No mitigations generated (LLM unavailable)._")
    for m in report.mitigations:
        lines += [
            f"- **[{m.priority.upper()}] {m.category_id}** — {m.issue}",
            f"  - _Fix:_ {m.suggestion}",
        ]

    return "\n".join(lines) + "\n"


def render_comparison_markdown(cmp: RunComparison) -> str:
    """Render a before/after regression comparison as Markdown."""
    arrow = "▲" if cmp.overall_delta > 0 else ("▼" if cmp.overall_delta < 0 else "▬")
    lines: list[str] = [
        "# Regression Comparison",
        "",
        f"**Before:** {cmp.before_target}  →  **After:** {cmp.after_target}",
        "",
        f"## Overall: {cmp.overall_before} → {cmp.overall_after} "
        f"({arrow} {cmp.overall_delta:+})",
        "",
        f"- Categories improved: **{cmp.num_improved}**, "
        f"regressed: **{cmp.num_regressed}**, unchanged: **{cmp.num_unchanged}**",
        "",
        "| Category | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for d in cmp.category_deltas:
        before = "—" if d.before_score is None else f"{d.before_score:.1f}"
        after = "—" if d.after_score is None else f"{d.after_score:.1f}"
        if d.delta is None:
            delta = "—"
        else:
            mark = "▲" if d.delta > 0 else ("▼" if d.delta < 0 else "▬")
            delta = f"{mark} {d.delta:+}"
        lines.append(f"| {d.name} | {before} | {after} | {delta} |")

    return "\n".join(lines) + "\n"


def save_report(
    report: RobustnessReport, *, base_dir: Path | str = DEFAULT_REPORTS_DIR
) -> tuple[Path, Path]:
    """Persist the report as JSON + Markdown. Returns (json_path, md_path)."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    stamp = report.created_at.replace(":", "").replace("-", "").replace(".", "")[:15]
    slug = _slugify(report.target_name)
    stem = f"{slug}-{stamp}"

    json_path = base / f"{stem}.json"
    md_path = base / f"{stem}.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _slugify(text: str) -> str:
    """Filesystem-safe slug for filenames."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower())
    return slug.strip("-") or "report"
