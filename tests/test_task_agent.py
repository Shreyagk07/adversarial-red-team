"""Tests for the Task agent layer (Phase 1).

All tests here are offline and key-free: they exercise the interface contract,
the offline Echo target, and the factory's error handling. This keeps CI green
without secrets while still locking down the behavior later phases rely on.
"""

from __future__ import annotations

import pytest

from agents.base import AgentResponse, EchoTaskAgent, TaskAgent
from agents.llm import (
    GEMINI_DEFAULT_MODEL,
    GROQ_DEFAULT_MODEL,
    MissingAPIKeyError,
    build_chat_model,
    default_model_for,
)
from backend.config import Settings


def test_echo_agent_satisfies_interface() -> None:
    """EchoTaskAgent is a TaskAgent and returns a well-formed response."""
    agent = EchoTaskAgent()
    assert isinstance(agent, TaskAgent)

    response = agent.answer("hello")
    assert isinstance(response, AgentResponse)
    assert response.text == "You said: hello"
    assert response.ok is True
    assert response.provider == "local"
    assert response.model == "echo"


def test_default_model_for_each_provider() -> None:
    """The factory maps each provider to its pinned default model id."""
    assert default_model_for("groq") == GROQ_DEFAULT_MODEL
    assert default_model_for("gemini") == GEMINI_DEFAULT_MODEL


def test_build_chat_model_missing_groq_key_is_actionable() -> None:
    """Building a Groq model without a key raises a clear, actionable error."""
    settings = Settings(llm_provider="groq", groq_api_key=None)
    with pytest.raises(MissingAPIKeyError) as exc_info:
        build_chat_model(settings)
    # The message should tell the user exactly where to get a key.
    assert "console.groq.com" in str(exc_info.value)


def test_build_chat_model_missing_gemini_key_is_actionable() -> None:
    """Building a Gemini model without a key raises a clear, actionable error."""
    settings = Settings(llm_provider="gemini", gemini_api_key=None)
    with pytest.raises(MissingAPIKeyError) as exc_info:
        build_chat_model(settings, provider="gemini")
    assert "aistudio.google.com" in str(exc_info.value)


def test_unknown_provider_rejected() -> None:
    """An unknown provider id is a hard error, not a silent default."""
    with pytest.raises(ValueError):
        default_model_for("openai")  # type: ignore[arg-type]
