"""Persistence for benchmark sweeps."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from crt_agent.config import Settings
from crt_agent.config import settings as default_settings
from crt_agent.schemas import Answer, Item
from crt_agent.store.models import Attempt, Base, BenchmarkRun, TraceStepRow


class Repository:
    def __init__(self, url: str | None = None, settings: Settings | None = None) -> None:
        cfg = settings or default_settings
        self.url = url or cfg.database_url
        self.engine = create_engine(self.url, future=True)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session()

    def save_run(
        self,
        *,
        provider: str,
        model: str,
        seed: int,
        agents: list[str],
        results: Iterable[tuple[Item, Answer]],
        notes: str = "",
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or uuid.uuid4().hex[:12]
        with self.session() as s:
            s.add(
                BenchmarkRun(
                    run_id=run_id,
                    provider=provider,
                    model=model,
                    seed=seed,
                    agents=agents,
                    notes=notes,
                )
            )
            for item, answer in results:
                attempt = Attempt(
                    run_id=run_id,
                    trace_id=answer.trace.trace_id,
                    agent=answer.agent,
                    item_id=item.item_id,
                    family=item.family,
                    item_set=item.item_set,
                    question=item.text,
                    expected=item.answer,
                    lure=item.lure,
                    value=answer.value,
                    intuitive_value=answer.intuitive_value,
                    correct=item.is_correct(answer.value),
                    hit_lure=item.is_lure(answer.value),
                    abstained=answer.abstained,
                    conflict_detected=answer.conflict_detected,
                    audit_valid=bool(answer.audit and answer.audit.valid),
                    audit_applicable=bool(answer.audit and answer.audit.applicable),
                    audit_reason=answer.audit.reason if answer.audit else "",
                    error=answer.error,
                    latency_ms=answer.latency_ms,
                    tokens_in=answer.tokens_in,
                    tokens_out=answer.tokens_out,
                )
                attempt.steps = [
                    TraceStepRow(
                        step_index=step.index,
                        kind=step.kind.value,
                        node=step.node,
                        rationale=step.rationale,
                        payload=step.payload,
                        duration_ms=step.duration_ms,
                    )
                    for step in answer.trace.steps
                ]
                s.add(attempt)
            s.commit()
        return run_id

    def latest_run_id(self) -> str | None:
        with self.session() as s:
            row = s.execute(
                select(BenchmarkRun.run_id).order_by(BenchmarkRun.id.desc()).limit(1)
            ).first()
            return row[0] if row else None

    def attempts(self, run_id: str) -> list[Attempt]:
        with self.session() as s:
            return list(s.scalars(select(Attempt).where(Attempt.run_id == run_id)))
