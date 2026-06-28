"""The adversarial loop, orchestrated as a LangGraph state graph.

This wires the two agents we already have into a single, inspectable pipeline:

    START -> challenge -> run_target -> END
              (generate)   (probe the target, record transcripts)

Why a graph for something this linear? Three reasons that pay off in later
phases: (1) the Judge becomes just one more node we splice in (Phase 4);
(2) LangGraph gives us a uniform place to hang tracing/observability (Phase 9);
(3) state is explicit and serializable, which we'll lean on for persistence.

The loop is dependency-injected with a target and a challenger, so the whole
graph can be exercised offline with a fake challenger + the EchoTaskAgent.
"""

from __future__ import annotations

from typing import Protocol

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.base import TaskAgent
from agents.categories import RobustnessCategory
from agents.schemas import CategoryScore, Exchange, LoopResult, TestCase, Verdict

# Default number of probes per run when the caller doesn't specify.
DEFAULT_NUM_TESTS = 5


class SupportsChallenge(Protocol):
    """Minimal interface the loop needs from a Challenger.

    Using a Protocol (instead of importing the concrete ChallengerAgent) keeps
    the loop decoupled and trivially fakeable in tests.
    """

    category: RobustnessCategory

    def generate(self, target_description: str, n: int) -> list[TestCase]: ...


class SupportsJudge(Protocol):
    """Minimal interface the loop needs from a Judge."""

    def evaluate(self, exchange: Exchange, category: RobustnessCategory) -> Verdict: ...


class LoopState(TypedDict):
    """State threaded through the graph.

    Each node reads what it needs and writes back a partial update. Keys are
    written at most once per run (challenge writes ``test_cases``; run_target
    writes ``exchanges``; judge rewrites ``exchanges`` with verdicts attached
    and writes ``category_score``), so plain assignment is correct and we don't
    need ``Annotated`` reducers yet.
    """

    target_description: str
    n_tests: int
    test_cases: list[TestCase]
    exchanges: list[Exchange]
    category_score: CategoryScore | None


class AdversarialLoop:
    """Runs Challenger -> Target (-> Judge) as a compiled LangGraph graph.

    The Judge is optional: pass one to score the run, or omit it to just
    collect raw transcripts (useful for debugging the target/challenger alone).
    """

    def __init__(
        self,
        target: TaskAgent,
        challenger: SupportsChallenge,
        judge: SupportsJudge | None = None,
    ) -> None:
        self.target = target
        self.challenger = challenger
        self.judge = judge
        self._graph = self._build_graph()

    # --- graph construction -------------------------------------------------
    def _build_graph(self):
        """Build and compile the state graph (closures capture the agents)."""

        def challenge(state: LoopState) -> dict[str, list[TestCase]]:
            """Generate the batch of probes for this run."""
            cases = self.challenger.generate(
                state["target_description"], state["n_tests"]
            )
            return {"test_cases": cases}

        def run_target(state: LoopState) -> dict[str, list[Exchange]]:
            """Send each probe to the target and record the exchange.

            Target errors are already captured inside AgentResponse (the target
            never raises), so a single bad answer can't break the loop.
            """
            exchanges = [
                Exchange(test_case=case, response=self.target.answer(case.prompt))
                for case in state["test_cases"]
            ]
            return {"exchanges": exchanges}

        def judge_node(state: LoopState) -> dict[str, object]:
            """Score every exchange and roll the verdicts into a category score.

            Rewrites ``exchanges`` with a verdict attached to each, and writes
            the aggregate ``category_score``.
            """
            assert self.judge is not None  # node only added when judge present
            category = self.challenger.category

            judged: list[Exchange] = [
                ex.model_copy(update={"verdict": self.judge.evaluate(ex, category)})
                for ex in state["exchanges"]
            ]
            score = CategoryScore.from_exchanges(category.id, judged)
            return {"exchanges": judged, "category_score": score}

        builder = StateGraph(LoopState)
        builder.add_node("challenge", challenge)
        builder.add_node("run_target", run_target)
        builder.add_edge(START, "challenge")
        builder.add_edge("challenge", "run_target")

        if self.judge is not None:
            # challenge -> run_target -> judge -> END
            builder.add_node("judge", judge_node)
            builder.add_edge("run_target", "judge")
            builder.add_edge("judge", END)
        else:
            # No judge: stop after collecting raw transcripts.
            builder.add_edge("run_target", END)

        return builder.compile()

    # --- public API ---------------------------------------------------------
    def run(self, n_tests: int = DEFAULT_NUM_TESTS) -> LoopResult:
        """Execute one full adversarial loop and return its transcripts."""
        initial: LoopState = {
            "target_description": self.target.description,
            "n_tests": n_tests,
            "test_cases": [],
            "exchanges": [],
            "category_score": None,
        }
        final: LoopState = self._graph.invoke(initial)

        return LoopResult(
            target_name=self.target.name,
            target_description=self.target.description,
            category_id=self.challenger.category.id,
            exchanges=final["exchanges"],
            category_score=final.get("category_score"),
        )
