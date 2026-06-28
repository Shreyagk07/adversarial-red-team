"""API request/response schemas (the HTTP contract).

Kept separate from the ORM models (storage/models.py) and the agent schemas
(agents/schemas.py): the wire format is its own concern and shouldn't leak ORM
or internal details. ``from_attributes=True`` lets us build these directly from
ORM rows.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CategoryOut(BaseModel):
    """A robustness category exposed to clients (e.g. the dashboard picker)."""

    id: str
    name: str
    description: str


class TargetCreate(BaseModel):
    """Body for registering a new target."""

    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    system_prompt: str = ""
    provider: str = "groq"
    model: str | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class TargetOut(BaseModel):
    """A target as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    system_prompt: str
    provider: str
    model: str | None
    temperature: float
    created_at: str


class EvaluateRequest(BaseModel):
    """Body for launching an evaluation run."""

    tests_per_category: int = Field(default=3, ge=1, le=20)
    # Optional subset of category ids; omit/empty for the full catalog.
    category_ids: list[str] | None = None


class RunOut(BaseModel):
    """A run summary (status + headline metrics)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    target_id: str
    status: str
    error: str | None
    tests_per_category: int
    overall_score: float | None
    overall_pass_rate: float | None
    total_tests: int | None
    total_passed: int | None
    total_failed: int | None
    created_at: str
