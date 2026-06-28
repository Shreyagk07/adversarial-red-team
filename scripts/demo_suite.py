"""Manual smoke test for the full multi-category suite (Phase 5).

Run after configuring an API key in .env:

    python -m scripts.demo_suite            # all categories, 3 probes each
    python -m scripts.demo_suite 2          # all categories, 2 probes each

Builds the demo target and runs every robustness category, then prints a
per-category score table and the overall robustness score.

NOTE: a full run makes roughly (categories x probes x ~2) + categories LLM
calls. With 6 categories and 3 probes that's ~60 calls — fine on Groq's free
tier but it takes a minute or two. Pass a smaller probe count to go faster.
"""

from __future__ import annotations

import sys

from agents.evaluator import RedTeamEvaluator
from agents.llm import MissingAPIKeyError
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
        print("Add a key to .env, then re-run. (The suite needs live LLMs.)")
        return

    print(f"Evaluating target '{target.name}' across "
          f"{len(evaluator.categories)} categories "
          f"({tests_per_category} probes each)...\n")

    suite = evaluator.run(tests_per_category=tests_per_category)

    # Per-category table.
    print(f"{'Category':<28}{'Score':>7}{'Passed':>10}")
    print("-" * 45)
    for result in suite.category_results:
        s = result.category_score
        if s is None:
            continue
        passed = f"{s.num_passed}/{s.num_tests}"
        print(f"{result.category_id:<28}{s.score:>7.1f}{passed:>10}")
    print("-" * 45)
    print(f"{'OVERALL':<28}{suite.overall_score:>7.1f}"
          f"{f'{suite.total_passed}/{suite.total_tests}':>10}")
    print(f"\nOverall robustness score: {suite.overall_score}/100 "
          f"(pass rate {suite.overall_pass_rate:.0%})")


if __name__ == "__main__":
    main()
