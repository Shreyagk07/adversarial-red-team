"""Manual smoke test for the Task agent (Phase 1).

Run this after putting a GROQ_API_KEY (or GEMINI_API_KEY) in your .env:

    python -m scripts.demo_task_agent

It builds the demo LLM target and asks a few prompts that probe the kinds of
weakness we'll later evaluate automatically (a clear factual question, an
ambiguous one, and a reasoning trap). For each it prints the answer plus
latency and token usage. With no key configured it falls back to the offline
EchoTaskAgent and tells you so.
"""

from __future__ import annotations

from agents.base import AgentResponse, EchoTaskAgent, TaskAgent
from agents.llm import MissingAPIKeyError
from agents.task_agent import LLMTaskAgent
from backend.config import get_settings

# A small, varied prompt set. Deliberately includes an ambiguous prompt and a
# classic reasoning trap to preview what the Challenger will probe in Phase 2.
DEMO_PROMPTS: list[str] = [
    "What is the capital of Australia?",
    "Is it better?",  # ambiguous: better than what?
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
    "ball. How much does the ball cost?",
]


def build_target() -> TaskAgent:
    """Build the LLM target, falling back to Echo if no key is configured."""
    settings = get_settings()
    try:
        target = LLMTaskAgent(settings)
        print(f"Using LLM target: provider={settings.llm_provider}\n")
        return target
    except MissingAPIKeyError as exc:
        print(f"[no API key] {exc}\n--> Falling back to offline EchoTaskAgent.\n")
        return EchoTaskAgent()


def main() -> None:
    target = build_target()
    print(f"Target: {target.name} — {target.description}\n" + "=" * 70)

    for i, prompt in enumerate(DEMO_PROMPTS, start=1):
        response: AgentResponse = target.answer(prompt)
        print(f"\n[{i}] PROMPT: {prompt}")
        print(f"    ANSWER: {response.text}")
        meta = f"    ({response.provider}/{response.model}, {response.latency_ms:.0f} ms"
        if response.output_tokens is not None:
            meta += f", {response.input_tokens}->{response.output_tokens} tok"
        meta += ")"
        print(meta)
        if not response.ok:
            print(f"    ERROR: {response.error}")

    print("\n" + "=" * 70)
    print("Done. (The ambiguous and trap prompts preview Phase 2's probes.)")


if __name__ == "__main__":
    main()
