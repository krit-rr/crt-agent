"""Postgres schema for benchmark runs.

Three tables, one per grain:

    benchmark_runs  — one sweep (which agents, which items, which model, when)
    attempts        — one agent x one item
    trace_steps     — one node execution

Storing steps as rows rather than a JSON blob is the point. Once they're rows you can
ask the questions that matter across a whole sweep in SQL — "which equations does the
parser emit when it gets bat/ball wrong", "how often does verify fail", "what's the
p95 latency of the parse node" — without re-running anything.

SQLite works for local development; Postgres is what the compose file brings up.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    provider: Mapped[str] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(128))
    seed: Mapped[int] = mapped_column(Integer)
    agents: Mapped[list] = mapped_column(JSON)
    notes: Mapped[str] = mapped_column(Text, default="")

    attempts: Mapped[list[Attempt]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("benchmark_runs.run_id"), index=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)

    agent: Mapped[str] = mapped_column(String(64), index=True)
    item_id: Mapped[str] = mapped_column(String(128), index=True)
    family: Mapped[str] = mapped_column(String(32), index=True)
    item_set: Mapped[str] = mapped_column(String(16), index=True)
    question: Mapped[str] = mapped_column(Text)

    expected: Mapped[float] = mapped_column(Float)
    lure: Mapped[float] = mapped_column(Float)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    intuitive_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    correct: Mapped[bool] = mapped_column(Boolean, index=True)
    hit_lure: Mapped[bool] = mapped_column(Boolean, index=True)
    abstained: Mapped[bool] = mapped_column(Boolean)
    conflict_detected: Mapped[bool] = mapped_column(Boolean)

    audit_valid: Mapped[bool] = mapped_column(Boolean)
    audit_applicable: Mapped[bool] = mapped_column(Boolean)
    audit_reason: Mapped[str] = mapped_column(Text, default="")

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped[BenchmarkRun] = relationship(back_populates="attempts")
    steps: Mapped[list[TraceStepRow]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )


Index("ix_attempts_agent_itemset", Attempt.agent, Attempt.item_set)


class TraceStepRow(Base):
    __tablename__ = "trace_steps"

    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"), index=True)
    step_index: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    node: Mapped[str] = mapped_column(String(32))
    rationale: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)

    attempt: Mapped[Attempt] = relationship(back_populates="steps")
