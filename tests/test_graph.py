"""Tests for the adversarial loop (Phase 3).

These run the *real* compiled LangGraph graph, but with offline agents: a fake
Challenger returning canned probes and the deterministic EchoTaskAgent. So we
verify the graph's wiring (challenge -> run_target, transcript assembly) with
no network and no API key.
"""

from __future__ import annotations

from agents.base import EchoTaskAgent
from agents.categories import AMBIGUITY
from agents.graph import AdversarialLoop
from agents.schemas import LoopResult, TestCase


class _FakeChallenger:
    """Returns a fixed set of probes; mimics the Challenger interface."""

    def __init__(self, n: int = 3) -> None:
        self.category = AMBIGUITY
        self._n = n
        self.received_description: str | None = None
        self.received_n: int | None = None

    def generate(self, target_description: str, n: int) -> list[TestCase]:
        # Record args so we can assert the loop passes the target through.
        self.received_description = target_description
        self.received_n = n
        return [
            TestCase(
                id=f"ambiguity-{i:03d}",
                category_id="ambiguity",
                probe_type="missing referent",
                prompt=f"Is it better {i}?",
                rationale="no referent",
                expected_behavior="ask what 'it' refers to",
            )
            for i in range(1, self._n + 1)
        ]


def test_loop_produces_one_exchange_per_probe() -> None:
    target = EchoTaskAgent()
    challenger = _FakeChallenger(n=3)
    loop = AdversarialLoop(target=target, challenger=challenger)

    result = loop.run(n_tests=3)

    assert isinstance(result, LoopResult)
    assert result.num_exchanges == 3
    assert result.category_id == "ambiguity"
    assert result.target_name == "echo-agent"


def test_loop_pairs_each_probe_with_its_response() -> None:
    target = EchoTaskAgent()
    challenger = _FakeChallenger(n=2)
    loop = AdversarialLoop(target=target, challenger=challenger)

    result = loop.run(n_tests=2)

    for ex in result.exchanges:
        # Echo target echoes the probe text, proving prompt->response routing.
        assert ex.response.text == f"You said: {ex.test_case.prompt}"
        assert ex.response.ok is True


def test_loop_passes_target_description_and_count_to_challenger() -> None:
    target = EchoTaskAgent()
    challenger = _FakeChallenger(n=4)
    loop = AdversarialLoop(target=target, challenger=challenger)

    loop.run(n_tests=4)

    # The loop should hand the target's own description to the challenger.
    assert challenger.received_description == target.description
    assert challenger.received_n == 4
