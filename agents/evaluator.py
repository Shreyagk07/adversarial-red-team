"""The suite evaluator — runs the adversarial loop across many categories.

Phase 4 scored a single category. This orchestrates the whole suite: for each
configured robustness category it builds a category-specific Challenger, runs
the (shared) target and Judge through the adversarial loop, and finally rolls
every category score up into one overall robustness number.

Dependency injection again does the heavy lifting:
  * ``target`` and ``judge`` are shared across all categories.
  * ``challenger_factory`` builds a Challenger for a given category — real in
    production, fake in tests — so the entire suite runs offline in CI.
"""

from __future__ import annotations

from typing import Callable

from agents.base import TaskAgent
from agents.categories import RobustnessCategory, list_categories
from agents.challenger import ChallengerAgent
from agents.graph import AdversarialLoop, SupportsChallenge, SupportsJudge
from agents.judge import JudgeAgent
from agents.schemas import LoopResult, SuiteResult
from backend.config import Settings, get_settings

# A function that produces a Challenger bound to a specific category.
ChallengerFactory = Callable[[RobustnessCategory], SupportsChallenge]

# Default probes generated per category in a suite run.
DEFAULT_TESTS_PER_CATEGORY = 5


class RedTeamEvaluator:
    """Runs a multi-category robustness evaluation against a target."""

    def __init__(
        self,
        target: TaskAgent,
        judge: SupportsJudge,
        challenger_factory: ChallengerFactory,
        categories: list[RobustnessCategory] | None = None,
    ) -> None:
        self.target = target
        self.judge = judge
        self.challenger_factory = challenger_factory
        # Default to the full catalog, in registration order.
        self.categories = categories if categories is not None else list_categories()

    def run(
        self, tests_per_category: int = DEFAULT_TESTS_PER_CATEGORY
    ) -> SuiteResult:
        """Run every configured category and aggregate an overall score."""
        results: list[LoopResult] = []
        for category in self.categories:
            challenger = self.challenger_factory(category)
            loop = AdversarialLoop(
                target=self.target, challenger=challenger, judge=self.judge
            )
            results.append(loop.run(n_tests=tests_per_category))

        return SuiteResult.from_results(
            target_name=self.target.name,
            target_description=self.target.description,
            results=results,
        )

    @classmethod
    def from_settings(
        cls,
        target: TaskAgent,
        settings: Settings | None = None,
        *,
        categories: list[RobustnessCategory] | None = None,
    ) -> "RedTeamEvaluator":
        """Convenience builder wiring real Challenger + Judge from settings.

        Constructs the Judge once and provides a factory that builds a real
        ChallengerAgent per category. Raises (via the underlying agents) if the
        selected provider has no API key configured.
        """
        settings = settings or get_settings()
        judge = JudgeAgent(settings)

        def factory(category: RobustnessCategory) -> SupportsChallenge:
            return ChallengerAgent(settings, category=category)

        return cls(target, judge, factory, categories)
