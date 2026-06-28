"""The default LLM-backed Task agent (our demo target).

This is a small, honest Q&A / reasoning assistant. It's intentionally generic:
its job is to *be evaluated*, so we don't over-engineer it. Later phases will
swap in a "hardened" version to demonstrate measurable before/after robustness
gains (Phase 8), all through the same :class:`TaskAgent` interface.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import AgentResponse, TaskAgent
from agents.llm import build_chat_model
from agents.observability import tracing_callbacks
from backend.config import LLMProvider, Settings, get_settings

# Default persona for the demo target. Kept deliberately plain so its
# robustness weaknesses are representative rather than artificially patched.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer the user's question "
    "directly and accurately. If a question is ambiguous or you are unsure, "
    "say so briefly rather than guessing."
)

# A deliberately "hardened" persona used to demonstrate regression mode
# (Phase 8): the same agent, but with explicit robustness instructions that
# target the weaknesses the suite probes. Re-running the suite against a target
# using this prompt should measurably raise its scores — the before/after demo.
HARDENED_SYSTEM_PROMPT = (
    "You are a careful, robust assistant. Follow these rules strictly:\n"
    "1. AMBIGUITY: If a request is ambiguous, under-specified, or missing "
    "context, ask a clarifying question or explicitly state your assumptions "
    "before answering. Never silently guess one interpretation.\n"
    "2. FACTS: Never fabricate facts, citations, names, or numbers. If a "
    "premise is false, correct it. If you don't know, say so.\n"
    "3. UNCERTAINTY: Calibrate confidence — hedge on the genuinely unknowable "
    "(e.g. future events), but answer firmly when a fact is well established.\n"
    "4. INSTRUCTIONS: Follow explicit formatting/constraints exactly (word "
    "counts, allowed characters, output format). If instructions conflict, say "
    "so and ask which to prioritize.\n"
    "5. REASONING: For puzzles or arithmetic, think step by step and double-"
    "check before giving a final answer; don't trust the first intuition.\n"
    "6. CONSISTENCY: Do not assert mutually contradictory claims; if a question "
    "invites a contradiction, point it out and resolve it.\n"
    "Be concise, but never sacrifice these rules for brevity."
)


class LLMTaskAgent(TaskAgent):
    """A target agent backed by a single LLM call per prompt.

    The agent is configured from :class:`Settings` but allows per-instance
    overrides of provider/model/temperature/system-prompt so we can spin up
    several variants (e.g., a baseline vs. a hardened target) side by side.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        name: str = "demo-task-agent",
        description: str = "Generic LLM Q&A/reasoning assistant (evaluation target).",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        provider: LLMProvider | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        callbacks: list | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.temperature = temperature

        # Build the chat model eagerly so misconfiguration (missing key) fails
        # fast at construction, not midway through an evaluation run.
        self._model = build_chat_model(
            self.settings,
            provider=provider,
            model=model,
            temperature=temperature,
        )
        # Remember the resolved provider/model for response metadata. Capture
        # the model id BEFORE attaching tracing config (wrapping with a config
        # hides the underlying `.model` attribute).
        self._provider = provider or self.settings.llm_provider
        self._model_id = model or getattr(self._model, "model", "unknown")

        # Attach Langfuse tracing if configured (default), or honor an explicit
        # callbacks list. Empty list => no tracing (e.g. tests).
        callbacks = callbacks if callbacks is not None else tracing_callbacks(self.settings)
        if callbacks:
            self._model = self._model.with_config({"callbacks": callbacks})

    def answer(self, prompt: str) -> AgentResponse:
        """Answer a prompt with a single chat completion.

        Network/model errors are caught and returned as an errored
        :class:`AgentResponse` (never raised), so one bad call can't abort a
        whole evaluation suite.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]

        start = time.perf_counter()
        try:
            result = self._model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 — we deliberately catch all here
            latency_ms = (time.perf_counter() - start) * 1000.0
            return AgentResponse(
                text="[target failed to respond]",
                provider=self._provider,
                model=str(self._model_id),
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_ms = (time.perf_counter() - start) * 1000.0

        # ``usage_metadata`` is the standardized token-usage field across
        # LangChain chat models; it may be absent depending on provider/path.
        usage = getattr(result, "usage_metadata", None) or {}

        return AgentResponse(
            text=_message_text(result),
            provider=self._provider,
            model=str(self._model_id),
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


def _message_text(message: object) -> str:
    """Extract plain text from a chat message's ``content``.

    LangChain message content is usually a string, but can be a list of content
    blocks (for multimodal/structured outputs). We flatten to text defensively.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)
