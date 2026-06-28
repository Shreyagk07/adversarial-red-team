"""Application configuration.

This module centralizes all runtime configuration and is the single seam
through which the rest of the app learns about the environment (API keys,
the chosen LLM provider, the app version, etc.).

We use ``pydantic-settings`` so that:
  * values are read from environment variables / a local ``.env`` file,
  * values are validated and type-coerced,
  * secrets never need to be hard-coded anywhere in the codebase.

Nothing in this file makes a network call or requires a key to be present.
That lets Phase 0 run with an empty ``.env`` while still establishing the
exact place where Phase 1 will plug real LLM credentials in.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The provider we default to for LLM calls. Groq's free tier is fast and
# generous, which suits the high call volume of an adversarial eval loop.
LLMProvider = Literal["groq", "gemini"]


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment.

    Attributes are populated (in priority order) from real environment
    variables first, then from a local ``.env`` file. Unknown variables in
    the environment are ignored so this stays robust as we add config later.
    """

    # --- App metadata -------------------------------------------------------
    app_name: str = "Adversarial Multi-Agent Red-Team System"
    app_version: str = "0.1.0"
    environment: Literal["dev", "staging", "prod"] = "dev"

    # --- LLM provider selection --------------------------------------------
    # Which provider the agents should use by default. Overridable via env.
    llm_provider: LLMProvider = "groq"

    # API keys. These are intentionally optional in Phase 0 so the app boots
    # without credentials; the agents added in later phases will require the
    # key for whichever provider is selected.
    groq_api_key: str | None = Field(default=None)
    gemini_api_key: str | None = Field(default=None)

    # --- Persistence --------------------------------------------------------
    # SQLAlchemy database URL. Defaults to a local SQLite file (zero setup).
    # THIS IS THE POSTGRES SEAM: point DATABASE_URL at a Postgres instance
    # (e.g. Neon) and nothing else in the codebase changes.
    database_url: str = "sqlite:///storage/redteam.db"

    # --- Observability (Langfuse) -------------------------------------------
    # Optional. When both keys are present, every agent LLM call is traced
    # (tokens, latency, cost). Absent keys => tracing is silently disabled.
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        """True when Langfuse credentials are configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # Pydantic-settings configuration: read from .env, ignore extra keys.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def active_provider_key_present(self) -> bool:
        """True if a key for the currently selected provider is configured.

        Health checks and the agent factory use this to fail fast with a
        clear message instead of making a doomed network call.
        """
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the ``.env`` file is parsed once per process. Tests can clear
    the cache via ``get_settings.cache_clear()`` when they need to override
    environment variables.
    """
    return Settings()
