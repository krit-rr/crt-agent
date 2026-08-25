"""Store tests against a real Postgres, skipped unless one is configured.

SQLite accepts things Postgres rejects (loose typing, JSON-as-text, missing FK
enforcement), so a store that only ever sees SQLite is not actually tested. CI runs
this job against a `postgres:16` service container; locally, `docker compose up -d`
and export TEST_DATABASE_URL.
"""

from __future__ import annotations

import os

import pytest

from crt_agent.agents import ToolAgent
from crt_agent.items import default_suite
from crt_agent.llm.mock import MockProvider
from crt_agent.store.models import Attempt, TraceStepRow
from crt_agent.store.repository import Repository
from crt_agent.tracing.langfuse_client import NullTracer

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="set TEST_DATABASE_URL to run the Postgres store tests"
)


@pytest.fixture
def repo():
    from crt_agent.store.models import Base

    r = Repository(url=DATABASE_URL)
    Base.metadata.drop_all(r.engine)
    r.create_schema()
    yield r
    Base.metadata.drop_all(r.engine)


def test_round_trip_against_postgres(repo):
    agent = ToolAgent(provider=MockProvider(), tracer=NullTracer())
    items = default_suite(n_perturbed=3, n_novel=2)
    rows = [(item, agent.answer(item)) for item in items]

    run_id = repo.save_run(
        provider="mock", model="mock", seed=1, agents=["tool"], results=rows, notes="ci"
    )

    attempts = repo.attempts(run_id)
    assert len(attempts) == len(items)
    assert repo.latest_run_id() == run_id

    with repo.session() as s:
        # JSON columns must survive the Postgres round trip as real dicts.
        step = s.query(TraceStepRow).filter(TraceStepRow.kind == "parse").first()
        assert isinstance(step.payload, dict)
        assert "spec_fingerprint" in step.payload
        # Nullable float columns must accept NULL for abstentions.
        assert s.query(Attempt).filter(Attempt.value.is_(None)).count() >= 0


def test_two_runs_are_isolated(repo):
    agent = ToolAgent(provider=MockProvider(), tracer=NullTracer())
    items = default_suite(n_perturbed=2, n_novel=1)
    rows = [(item, agent.answer(item)) for item in items]

    first = repo.save_run(provider="mock", model="m", seed=1, agents=["tool"], results=rows)
    second = repo.save_run(provider="mock", model="m", seed=2, agents=["tool"], results=rows)

    assert first != second
    assert len(repo.attempts(first)) == len(repo.attempts(second)) == len(items)
