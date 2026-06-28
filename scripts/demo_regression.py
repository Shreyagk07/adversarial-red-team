"""Manual smoke test for regression mode (Phase 8) — the killer demo.

Run after configuring an API key in .env:

    python -m scripts.demo_regression          # 3 probes/category
    python -m scripts.demo_regression 2        # 2 probes/category

Evaluates the SAME suite against two targets:
  * BEFORE: the baseline demo agent (default system prompt).
  * AFTER:  a 'hardened' agent (robustness instructions in its system prompt).

Then prints a before/after comparison, demonstrating that the hardening
measurably improved the robustness scores.
"""

from __future__ import annotations

import sys

from agents.evaluator import RedTeamEvaluator
from agents.llm import MissingAPIKeyError
from agents.reporter import Reporter, render_comparison_markdown
from agents.schemas import RunComparison
from agents.task_agent import HARDENED_SYSTEM_PROMPT, LLMTaskAgent
from backend.config import get_settings


def _evaluate(target: LLMTaskAgent, n: int):
    """Run the full suite + report for one target."""
    settings = get_settings()
    evaluator = RedTeamEvaluator.from_settings(target, settings)
    suite = evaluator.run(tests_per_category=n)
    return Reporter(settings).generate(suite)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    settings = get_settings()

    try:
        baseline = LLMTaskAgent(settings, name="baseline-agent")
        hardened = LLMTaskAgent(
            settings, name="hardened-agent", system_prompt=HARDENED_SYSTEM_PROMPT
        )
    except MissingAPIKeyError as exc:
        print(f"[no API key] {exc}")
        print("Add a key to .env, then re-run. (Regression mode needs LLMs.)")
        return

    print(f"BEFORE: evaluating '{baseline.name}' ({n} probes/category)...")
    before_report = _evaluate(baseline, n)
    print(f"  overall: {before_report.overall_score}/100")

    print(f"AFTER:  evaluating '{hardened.name}' ({n} probes/category)...")
    after_report = _evaluate(hardened, n)
    print(f"  overall: {after_report.overall_score}/100\n")

    comparison = RunComparison.from_reports(before_report, after_report)
    print(render_comparison_markdown(comparison))


if __name__ == "__main__":
    main()
