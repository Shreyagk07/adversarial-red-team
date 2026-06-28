"""Manual smoke test for the full Challenger -> Target -> Judge loop (Phase 4).

Run after configuring an API key in .env:

    python -m scripts.demo_judge

Builds all three agents, runs one scored loop over the ambiguity category, and
prints each transcript with the Judge's verdict, then the rolled-up category
score. The Judge uses the larger model at temperature 0 so its grading is as
consistent as possible.
"""

from __future__ import annotations

from agents.categories import AMBIGUITY
from agents.challenger import ChallengerAgent
from agents.graph import AdversarialLoop
from agents.judge import JudgeAgent
from agents.llm import MissingAPIKeyError
from agents.task_agent import LLMTaskAgent
from backend.config import get_settings


def main() -> None:
    settings = get_settings()
    try:
        target = LLMTaskAgent(settings)
        challenger = ChallengerAgent(settings, category=AMBIGUITY)
        judge = JudgeAgent(settings)
    except MissingAPIKeyError as exc:
        print(f"[no API key] {exc}")
        print("Add a key to .env, then re-run. (The full loop needs live LLMs.)")
        return

    loop = AdversarialLoop(target=target, challenger=challenger, judge=judge)
    print(f"Scored loop: category={AMBIGUITY.id}, target={target.name}")
    print("=" * 74)

    result = loop.run(n_tests=5)

    for i, ex in enumerate(result.exchanges, start=1):
        v = ex.verdict
        mark = "PASS" if (v and v.passed) else "FAIL"
        sev = v.severity if v else "n/a"
        print(f"\n[{i}] {ex.test_case.id}  ->  {mark} (severity={sev})")
        print(f"  PROBE:  {ex.test_case.prompt}")
        print(f"  TARGET: {ex.response.text}")
        print(f"  JUDGE:  {v.reason if v else '(no verdict)'}")

    score = result.category_score
    print("\n" + "=" * 74)
    if score:
        print(
            f"Category '{score.category_id}': score={score.score}/100 | "
            f"passed {score.num_passed}/{score.num_tests} | "
            f"pass_rate={score.pass_rate} | severities={score.severity_counts}"
        )


if __name__ == "__main__":
    main()
