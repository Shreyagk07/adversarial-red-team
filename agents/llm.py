"""LLM factory — the one place we construct chat models.

Centralizing model construction means the rest of the codebase asks for "a
chat model for the task role" and never touches provider-specific classes or
keys. Swapping providers, changing default models, or adding a new backend is
a change in exactly this file.

Supported providers:
  * ``groq``   — default. Fast, generous free tier.
  * ``gemini`` — Google's free tier, used as a fallback.

We pin sensible *current* default model ids (verified against provider docs on
2026-06-28). Note that ``llama-3.3-70b-versatile`` was deprecation-announced by
Groq on 2026-06-17, so we default to the recommended ``openai/gpt-oss-*`` line.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import LLMProvider, Settings

# Per-provider default models. Overridable via Settings / env at the call site.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"      # fast + cheap; good for targets
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"      # Google free-tier workhorse


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider has no API key configured.

    Carries an actionable message (where to get the key) so failures during a
    run are self-explanatory instead of surfacing as opaque auth errors deep
    inside the provider SDK.
    """


def default_model_for(provider: LLMProvider) -> str:
    """Return the default model id for a provider."""
    if provider == "groq":
        return GROQ_DEFAULT_MODEL
    if provider == "gemini":
        return GEMINI_DEFAULT_MODEL
    raise ValueError(f"Unknown provider: {provider!r}")


def build_chat_model(
    settings: Settings,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> BaseChatModel:
    """Construct a LangChain chat model from settings.

    Args:
        settings: Application settings holding provider selection and keys.
        provider: Override the provider (defaults to ``settings.llm_provider``).
        model: Override the model id (defaults to the provider's default).
        temperature: Sampling temperature. Lower = more deterministic; we keep
            targets fairly low-variance so evaluations are reproducible.

    Returns:
        A ready-to-invoke :class:`BaseChatModel`.

    Raises:
        MissingAPIKeyError: If no key is configured for the chosen provider.
        ValueError: If the provider name is unknown.
    """
    provider = provider or settings.llm_provider

    if provider == "groq":
        if not settings.groq_api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Create a free key at "
                "https://console.groq.com/keys and add it to your .env file."
            )
        # Imported lazily so the package is only needed when actually used
        # (keeps the offline/echo path importable without provider SDKs).
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or GROQ_DEFAULT_MODEL,
            temperature=temperature,
            api_key=settings.groq_api_key,
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set. Create a free key at "
                "https://aistudio.google.com/app/apikey and add it to your .env."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or GEMINI_DEFAULT_MODEL,
            temperature=temperature,
            google_api_key=settings.gemini_api_key,
        )

    raise ValueError(f"Unknown provider: {provider!r}")
