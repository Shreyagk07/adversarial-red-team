"""Manual smoke test for the robustness report (Phase 6).

Run after configuring an API key in .env:

    python -m scripts.demo_report          # 3 probes/category
    python -m scripts.demo_report 2        # 2 probes/category

Runs the full suite, generates a robustness report (with LLM mitigations),
prints the Markdown to the console, and saves JSON + Markdown under
storage/reports/.
"""

from __future__ import annotations

import sys

from agents.evaluator import RedTeamEvaluator
from agents.llm import MissingAPIKeyError
from agents.reporter import Reporter, render_markdown, save_report
from agents.task_agent import LLMTaskAgent
from backend.config import get_settings


def main() -> None:
    tests_per_category = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    settings = get_settings()
    try:
        target = LLMTaskAgent(settings)
        evaluator = RedTeamEvaluator.from_settings(target, settings)
    except MissingAPIKeyError as exc:
        print(f"[no API key] {exc}")
        print("Add a key to .env, then re-run. (Report generation needs LLMs.)")
        return

    print(f"Running suite for '{target.name}' ({tests_per_category} probes/category)...")
    suite = evaluator.run(tests_per_category=tests_per_category)

    reporter = Reporter(settings)
    print(f"Generating report (mitigations enabled: {reporter.mitigations_enabled})...\n")
    report = reporter.generate(suite)

    print(render_markdown(report))

    json_path, md_path = save_report(report)
    print(f"\nSaved:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
