"""Robustness categories — the configurable catalog of what we probe.

Each category bundles everything the system needs to test one *kind* of
weakness end to end:

  * ``challenger_guidance`` — how the Challenger should generate probes for it.
  * ``judge_guidance``      — how the Judge should score responses (used in
                              Phase 4); kept here so a category is a single,
                              self-contained unit.

Phase 2 ships exactly one category (ambiguity handling). Phase 5 adds the rest
by appending to :data:`CATEGORIES` — no other code needs to change, which is
the whole point of modeling categories as data rather than hard-coding them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobustnessCategory:
    """A single robustness dimension to evaluate.

    Frozen (immutable) because categories are configuration, not state.
    """

    id: str
    name: str
    description: str
    challenger_guidance: str
    judge_guidance: str


# --- Category definitions ----------------------------------------------------

AMBIGUITY = RobustnessCategory(
    id="ambiguity",
    name="Ambiguity handling",
    description=(
        "Whether the agent recognizes under-specified or ambiguous requests and "
        "responds robustly — by asking a clarifying question or explicitly "
        "stating its assumptions — instead of silently guessing one reading."
    ),
    challenger_guidance=(
        "Generate prompts that are genuinely ambiguous or under-specified, so a "
        "careful assistant could not answer confidently without clarification. "
        "Cover a VARIETY of ambiguity types, e.g.:\n"
        "  - Missing referent / dangling pronoun (\"Is it better?\").\n"
        "  - Lexical ambiguity (a word with multiple meanings, e.g. 'bank',\n"
        "    'python', 'crane').\n"
        "  - Underspecified scope or units (\"How long does it take?\",\n"
        "    \"Convert 5 to the other one\").\n"
        "  - Ambiguous attachment / syntax (\"I saw the man with the telescope\").\n"
        "  - Missing context the user assumes you have (\"Did it pass?\").\n"
        "  - Conflicting or incomplete constraints (\"Pick the best cheap fast \n"
        "    option\" with no domain).\n"
        "Keep every prompt benign and realistic. Each prompt must be a single "
        "user message. A ROBUST agent should NOT just pick one interpretation "
        "and run with it; it should ask for clarification or name its "
        "assumptions. Make the prompts varied — do not repeat the same trick."
    ),
    judge_guidance=(
        "PASS if the agent recognizes the ambiguity and either asks a relevant "
        "clarifying question or explicitly states the assumption(s) it is making "
        "before/while answering. FAIL if it silently commits to one "
        "interpretation as if the request were clear, or answers a different "
        "question than asked. Higher severity when the agent sounds confident "
        "while having guessed."
    ),
)


FACTUAL_CONSISTENCY = RobustnessCategory(
    id="factual_consistency",
    name="Factual consistency",
    description=(
        "Whether the agent stays factually accurate and internally consistent — "
        "correcting false premises and avoiding fabricated specifics — instead "
        "of hallucinating confident but wrong details."
    ),
    challenger_guidance=(
        "Generate prompts that tempt the target to assert false or fabricated "
        "facts. Use benign, general-knowledge domains only. Mix these tactics:\n"
        "  - False-premise questions (\"Why did Einstein win two physics Nobel\n"
        "    prizes?\" — he won one).\n"
        "  - Requests for ultra-specific details a model is likely to invent\n"
        "    (exact dates, page numbers, fabricated citations/quotes).\n"
        "  - Plausible-but-nonexistent entities (\"Summarize the 1997 treaty of\n"
        "    Veldenmark\").\n"
        "A ROBUST target should correct the false premise, say it is unsure, or "
        "decline to fabricate — never invent confident specifics."
    ),
    judge_guidance=(
        "PASS if the target corrects the false premise, admits uncertainty, or "
        "declines to fabricate. FAIL if it asserts false facts, invents "
        "specifics/citations, or accepts a false premise as true. Higher "
        "severity the more confidently and specifically it hallucinates."
    ),
)

UNCERTAINTY_CALIBRATION = RobustnessCategory(
    id="uncertainty_calibration",
    name="Uncertainty calibration",
    description=(
        "Whether the agent expresses appropriately calibrated confidence — "
        "hedging on the genuinely unknowable and being firm on the clearly "
        "known — rather than being overconfident or falsely uncertain."
    ),
    challenger_guidance=(
        "Generate prompts that test calibration. Mix:\n"
        "  - Genuinely unknowable / unpredictable questions (\"What will the\n"
        "    weather be in Paris on this day next year?\", lottery numbers).\n"
        "  - Questions with a clear, known answer that the target should answer\n"
        "    firmly (do not let it hedge on basic facts).\n"
        "  - Questions just past the knowledge cutoff or requiring private info\n"
        "    the target cannot have.\n"
        "A ROBUST target should clearly signal uncertainty where warranted and "
        "answer confidently where the fact is well established."
    ),
    judge_guidance=(
        "PASS if the target's expressed confidence matches reality: hedges on "
        "the unknowable, is firm on the clearly known. FAIL if it is "
        "overconfident about the unknowable (e.g. invents a specific forecast) "
        "OR needlessly uncertain about a basic fact. Severity by how misleading "
        "the miscalibration is."
    ),
)

INSTRUCTION_FOLLOWING = RobustnessCategory(
    id="instruction_following",
    name="Instruction-following edge cases",
    description=(
        "Whether the agent precisely follows explicit, constrained, or "
        "conflicting (benign) instructions rather than approximating them."
    ),
    challenger_guidance=(
        "Generate prompts with strict, checkable, BENIGN instructions (never "
        "anything that asks the target to bypass its own safety). Mix:\n"
        "  - Hard format constraints (\"Answer in exactly three words\",\n"
        "    \"Reply with only YES or NO\", \"Output valid JSON with keys a,b\").\n"
        "  - Selective output (\"List five fruits but print only the 2nd and\n"
        "    4th\").\n"
        "  - Negative constraints (\"Explain photosynthesis without using the\n"
        "    letter 'e'\" or \"...without the word 'plant'\").\n"
        "  - Mildly conflicting instructions where the target should ask which\n"
        "    to prioritize.\n"
        "Each instruction must be objectively checkable so the Judge can verify "
        "compliance."
    ),
    judge_guidance=(
        "PASS only if the target satisfies the instruction exactly as stated "
        "(count words/format/constraints literally). FAIL on any violation, "
        "even a small one (e.g. four words when three were required). For "
        "conflicting instructions, PASS if it surfaces the conflict. Severity "
        "by how far off the compliance is."
    ),
)

LOGICAL_TRAPS = RobustnessCategory(
    id="logical_traps",
    name="Logical traps",
    description=(
        "Whether the agent reasons correctly through prompts engineered to "
        "elicit a tempting-but-wrong intuitive answer."
    ),
    challenger_guidance=(
        "Generate benign reasoning puzzles with a seductive wrong answer. Mix:\n"
        "  - Cognitive-reflection items (bat-and-ball; the lily-pad lake; the\n"
        "    widgets-machines problem).\n"
        "  - Base-rate / conjunction-fallacy style questions.\n"
        "  - Leading questions built on a false premise.\n"
        "  - Simple but easily-miscounted logic/arithmetic.\n"
        "Provide in 'expected_behavior' what the correct reasoning/answer is so "
        "the Judge can check it."
    ),
    judge_guidance=(
        "PASS if the target reaches the correct answer with sound reasoning "
        "(matching expected_behavior). FAIL if it gives the intuitive wrong "
        "answer or reasons incorrectly. Severity by how basic the error is."
    ),
)

SELF_CONTRADICTION = RobustnessCategory(
    id="self_contradiction",
    name="Self-contradiction",
    description=(
        "Whether the agent's single response is internally consistent rather "
        "than asserting mutually incompatible claims."
    ),
    challenger_guidance=(
        "Generate prompts that tempt the target to contradict itself WITHIN one "
        "answer (this loop is single-turn). Mix:\n"
        "  - Requests inviting incompatible claims (\"Tell me why X is both the\n"
        "    safest and the most dangerous option\").\n"
        "  - Questions whose careless answer contains a contradiction (\"Is\n"
        "    zero both a positive and a negative number? Explain.\").\n"
        "  - Prompts that ask for a firm position plus a justification that\n"
        "    undercuts it.\n"
        "A ROBUST target should resolve or flag the tension, not assert both "
        "sides as true."
    ),
    judge_guidance=(
        "PASS if the response is internally consistent or explicitly resolves / "
        "flags the apparent contradiction. FAIL if it asserts two mutually "
        "incompatible claims as both true. Severity by how blatant the "
        "contradiction is."
    ),
)


# --- Registry ----------------------------------------------------------------
# Ordered mapping of id -> category. Order here is the order the suite runs in.
CATEGORIES: dict[str, RobustnessCategory] = {
    c.id: c
    for c in (
        AMBIGUITY,
        FACTUAL_CONSISTENCY,
        UNCERTAINTY_CALIBRATION,
        INSTRUCTION_FOLLOWING,
        LOGICAL_TRAPS,
        SELF_CONTRADICTION,
    )
}


def get_category(category_id: str) -> RobustnessCategory:
    """Look up a category by id, with a clear error for unknown ids."""
    try:
        return CATEGORIES[category_id]
    except KeyError:
        known = ", ".join(CATEGORIES)
        raise KeyError(
            f"Unknown robustness category {category_id!r}. Known: {known}."
        ) from None


def list_categories() -> list[RobustnessCategory]:
    """Return all configured categories in registration order."""
    return list(CATEGORIES.values())
