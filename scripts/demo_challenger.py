"""Manual smoke test for the Challenger agent (Phase 2).

Run after configuring an API key in .env:

    python -m scripts.demo_challenger

It asks the Challenger to generate a batch of ambiguity-handling probes for our
demo target and prints each probe with its type, rationale, and the behavior a
robust target should exhibit. Eyeball the output: the prompts should be
genuinely ambiguous and varied (different ambiguity types, not five rewrites of
the same trick).
"""

from __future__ import annotations

from agents.categories import AMBIGUITY
from agents.challenger import ChallengerAgent
from agents.llm import MissingAPIKeyError
from backend.config import get_settings

# A description of the target the Challenger is probing. In later phases this
# comes from the registered target; for now we describe our demo agent.
TARGET_DESCRIPTION = (
    "A general-purpose Q&A/reasoning assistant that answers user questions "
    "concisely and directly."
)


def main() -> None:
    settings = get_settings()
    try:
        challenger = ChallengerAgent(settings, category=AMBIGUITY)
    except MissingAPIKeyError as exc:
        print(f"[no API key] {exc}")
        print("Add a key to .env, then re-run. (Generation needs a live LLM.)")
        return

    print(f"Category: {AMBIGUITY.name} ({AMBIGUITY.id})")
    print(f"Provider: {settings.llm_provider}\n" + "=" * 72)

    cases = challenger.generate(TARGET_DESCRIPTION, n=5)
    for case in cases:
        print(f"\n[{case.id}]  type={case.probe_type}  difficulty={case.difficulty}")
        print(f"  PROMPT:   {case.prompt}")
        print(f"  WHY:      {case.rationale}")
        print(f"  EXPECTED: {case.expected_behavior}")

    print("\n" + "=" * 72)
    print(f"Generated {len(cases)} probes. Next phase runs them against the target.")


if __name__ == "__main__":
    main()
