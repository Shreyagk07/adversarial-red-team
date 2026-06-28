"""ORM models — the persisted shape of targets, runs, and results.

Schema overview::

    targets   1───*  runs   1───*  case_results
                       │
                       ├──*  category_scores
                       └──*  mitigations

We store BOTH normalized rows (for history, filtering, and per-case drill-down)
AND the full report JSON on the run (for lossless reconstruction). The JSON
blob keeps us from having to re-join everything to rebuild a RobustnessReport,
while the normalized rows make the data queryable — a common, pragmatic split.

Type choices are deliberately portable across SQLite and Postgres (String,
Integer, Float, Boolean, Text). Timestamps are stored as ISO-8601 strings to
sidestep SQLite's lack of a native datetime type.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from storage.db import Base


class Target(Base):
    """A registered evaluation target (its configuration)."""

    __tablename__ = "targets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(String(32), default="groq")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.3)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)

    runs: Mapped[list["Run"]] = relationship(
        back_populates="target", cascade="all, delete-orphan"
    )


class Run(Base):
    """One evaluation run against a target."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_id: Mapped[str] = mapped_column(
        ForeignKey("targets.id", ondelete="CASCADE"), index=True
    )
    # 'running' -> 'completed' | 'failed'. Lets the API return immediately and
    # the dashboard poll for completion.
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    tests_per_category: Mapped[int] = mapped_column(Integer, default=5)

    # Headline metrics (null until the run completes).
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_pass_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tests: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_passed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_failed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Full RobustnessReport JSON for lossless reconstruction.
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    target: Mapped["Target"] = relationship(back_populates="runs")
    category_scores: Mapped[list["CategoryScoreRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    case_results: Mapped[list["CaseResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    mitigations: Mapped[list["MitigationRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class CategoryScoreRow(Base):
    """Per-category rollup for a run."""

    __tablename__ = "category_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    category_id: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float)
    num_tests: Mapped[int] = mapped_column(Integer)
    num_passed: Mapped[int] = mapped_column(Integer)
    num_failed: Mapped[int] = mapped_column(Integer)

    run: Mapped["Run"] = relationship(back_populates="category_scores")


class CaseResult(Base):
    """A single probe/response/verdict, persisted for drill-down."""

    __tablename__ = "case_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    category_id: Mapped[str] = mapped_column(String(64), index=True)
    test_case_id: Mapped[str] = mapped_column(String(64))
    probe_type: Mapped[str] = mapped_column(String(128), default="")
    prompt: Mapped[str] = mapped_column(Text)
    expected_behavior: Mapped[str] = mapped_column(Text, default="")
    response_text: Mapped[str] = mapped_column(Text, default="")
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="case_results")


class MitigationRow(Base):
    """A persisted hardening suggestion."""

    __tablename__ = "mitigations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    category_id: Mapped[str] = mapped_column(String(64))
    issue: Mapped[str] = mapped_column(Text)
    suggestion: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(String(16))

    run: Mapped["Run"] = relationship(back_populates="mitigations")
