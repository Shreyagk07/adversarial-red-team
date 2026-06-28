"""Core agent abstractions — the *pluggable target* interface.

The single most important design idea in this project is that the thing being
evaluated (the "Task agent" / target) is swappable. Today it's a small LLM
agent we build; tomorrow it could be any system behind an API. Everything
downstream (Challenger, Judge, report) depends only on this thin interface,
never on a concrete implementation.

A target is anything that can take a string prompt and return an
:class:`AgentResponse`. That's it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """A single answer produced by a target agent.

    We capture not just the text but enough metadata to make later judging,
    reporting, and observability meaningful (which model answered, how long it
    took, token usage when the provider reports it).
    """

    text: str = Field(description="The agent's natural-language answer.")
    provider: str = Field(description="LLM provider that produced the answer, e.g. 'groq'.")
    model: str = Field(description="Concrete model id, e.g. 'openai/gpt-oss-20b'.")
    latency_ms: float = Field(description="Wall-clock time for the call, milliseconds.")

    # Token usage is best-effort: not every provider/path reports it.
    input_tokens: int | None = Field(default=None)
    output_tokens: int | None = Field(default=None)

    # Set when the call failed and ``text`` holds a safe fallback message.
    error: str | None = Field(default=None)

    @property
    def ok(self) -> bool:
        """True when the call completed without an error."""
        return self.error is None


class TaskAgent(ABC):
    """Abstract base class for any target under evaluation.

    Implement this to plug a new target into the red-team system. The contract
    is deliberately tiny: a stable ``name``/``description`` for reporting, and
    an ``answer`` method that maps a prompt to an :class:`AgentResponse`.

    Implementations should *not* raise on ordinary model/network failures.
    Instead, return an :class:`AgentResponse` with ``error`` set and a safe
    fallback ``text`` — the evaluation must keep running even when the target
    misbehaves (a crashing target is itself a robustness finding, not a reason
    to abort the whole suite).
    """

    #: Human-readable identifier used in reports and dashboards.
    name: str = "unnamed-target"
    #: One-line description of what this target is/does.
    description: str = ""

    @abstractmethod
    def answer(self, prompt: str) -> AgentResponse:
        """Answer a single prompt and return a structured response."""
        raise NotImplementedError


class EchoTaskAgent(TaskAgent):
    """A trivial, deterministic target that needs no API key.

    Useful as (a) a reference implementation of the interface, (b) an offline
    target for unit tests and CI, and (c) a sanity baseline — an agent this
    dumb should fail lots of robustness probes, which is a nice smoke test of
    the *evaluator* later on.
    """

    name = "echo-agent"
    description = "Deterministic echo target (no LLM). For tests and offline runs."

    def answer(self, prompt: str) -> AgentResponse:
        return AgentResponse(
            text=f"You said: {prompt}",
            provider="local",
            model="echo",
            latency_ms=0.0,
        )
