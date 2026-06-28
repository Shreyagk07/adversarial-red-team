"""Manual smoke test for the adversarial loop (Phase 3).

Run after configuring an API key in .env:

    python -m scripts.demo_loop

Builds the real Challenger + demo Task target, runs one full loop over the
ambiguity category, and prints each transcript: the probe, the target's answer,
and (for reference) what a robust answer should look like. Eyeball whether the
target tends to guess instead of asking for clarification — that's the weakness
the Judge will quantify in Phase 4.
"""

from __future__ import annotations

from agents.categories import AMBIGUITY
from agents.challenger import ChallengerAgent
from agents.graph import AdversarialLoop
from agents.llm import MissingAPIKeyError
from agents.task_agent import LLMTaskAgent
from backend.config import get_settings


def main() -> None:
    settings = get_settings()
    try:
        target = LLMTaskAgent(settings)
        challenger = ChallengerAgent(settings, category=AMBIGUITY)
    except MissingAPIKeyError as exc:
        print(f"[no API key] {exc}")
        print("Add a key to .env, then re-run. (The loop needs live LLMs.)")
        return

    loop = AdversarialLoop(target=target, challenger=challenger)
    print(f"Running adversarial loop: category={AMBIGUITY.id}, target={target.name}")
    print("=" * 74)

    result = loop.run(n_tests=5)

    for i, ex in enumerate(result.exchanges, start=1):
        print(f"\n[{i}] {ex.test_case.id}  (type={ex.test_case.probe_type})")
        print(f"  PROBE:    {ex.test_case.prompt}")
        print(f"  TARGET:   {ex.response.text}")
        print(f"  EXPECTED: {ex.test_case.expected_behavior}")
        if not ex.response.ok:
            print(f"  ERROR:    {ex.response.error}")

    print("\n" + "=" * 74)
    print(
        f"Recorded {result.num_exchanges} exchanges for category "
        f"'{result.category_id}'. Phase 4 will score each one."
    )


if __name__ == "__main__":
    main()
