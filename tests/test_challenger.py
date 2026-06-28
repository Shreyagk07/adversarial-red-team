"""Tests for the Challenger agent and category catalog (Phase 2).

These are fully offline: we inject a fake chat model that returns canned
structured output, so we can assert the Challenger's *orchestration* logic
(prompt assembly, id assignment, category tagging, field mapping) without a
network call or API key.
"""

from __future__ import annotations

from agents.categories import (
    AMBIGUITY,
    CATEGORIES,
    get_category,
    list_categories,
)
from agents.challenger import ChallengerAgent
from agents.schemas import GeneratedTestCase, GeneratedTests, TestCase


# --- Fakes -------------------------------------------------------------------
class _FakeStructuredRunnable:
    """Stands in for ``model.with_structured_output(...)``'s return value."""

    def __init__(self, result: GeneratedTests) -> None:
        self._result = result
        self.last_messages: list[object] | None = None

    def invoke(self, messages: list[object]) -> GeneratedTests:
        # Record what it was called with so tests can assert prompt content.
        self.last_messages = messages
        return self._result


class _FakeChatModel:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, result: GeneratedTests) -> None:
        self._runnable = _FakeStructuredRunnable(result)

    def with_structured_output(self, schema: type) -> _FakeStructuredRunnable:
        return self._runnable


def _sample_generated(n: int = 3) -> GeneratedTests:
    return GeneratedTests(
        tests=[
            GeneratedTestCase(
                probe_type=f"type-{i}",
                prompt=f"ambiguous prompt {i}",
                rationale=f"why {i} is tricky",
                expected_behavior=f"ask for clarification {i}",
                difficulty="medium",
            )
            for i in range(1, n + 1)
        ]
    )


# --- Category catalog tests --------------------------------------------------
def test_ambiguity_category_registered() -> None:
    assert "ambiguity" in CATEGORIES
    assert get_category("ambiguity") is AMBIGUITY
    assert AMBIGUITY in list_categories()


def test_unknown_category_raises_clear_error() -> None:
    try:
        get_category("does-not-exist")
    except KeyError as exc:
        assert "Unknown robustness category" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected KeyError for unknown category")


# --- Challenger tests --------------------------------------------------------
def test_generate_assigns_ids_and_category() -> None:
    fake = _FakeChatModel(_sample_generated(3))
    challenger = ChallengerAgent(category=AMBIGUITY, chat_model=fake)

    cases = challenger.generate("a demo target", n=3)

    assert len(cases) == 3
    assert all(isinstance(c, TestCase) for c in cases)
    # Ids are stable, zero-padded, and namespaced by category.
    assert [c.id for c in cases] == ["ambiguity-001", "ambiguity-002", "ambiguity-003"]
    assert all(c.category_id == "ambiguity" for c in cases)
    # Fields are mapped through faithfully.
    assert cases[0].prompt == "ambiguous prompt 1"
    assert cases[0].expected_behavior == "ask for clarification 1"


def test_generate_embeds_category_and_target_in_prompt() -> None:
    fake = _FakeChatModel(_sample_generated(2))
    challenger = ChallengerAgent(category=AMBIGUITY, chat_model=fake)

    challenger.generate("My special target agent", n=2)

    # The system+user prompts should mention the category and the target.
    sent = fake._runnable.last_messages
    assert sent is not None
    combined = " ".join(getattr(m, "content", "") for m in sent)
    assert AMBIGUITY.name in combined
    assert "My special target agent" in combined
    assert "2" in combined  # the requested count appears in the instructions
