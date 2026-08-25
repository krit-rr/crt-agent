"""Sweep, metrics, persistence, and the CLI."""

from __future__ import annotations

import math

import pytest

from crt_agent.agents import AblationAgent, DualProcessAgent, SymbolicAgent, ToolAgent
from crt_agent.bench import failure_digest, render_answer, run_sweep, summary_table
from crt_agent.cli import main
from crt_agent.items import default_suite
from crt_agent.store.repository import Repository


@pytest.fixture(scope="module")
def sweep():
    from crt_agent.llm.mock import MockProvider
    from crt_agent.tracing.langfuse_client import NullTracer

    provider, tracer = MockProvider(), NullTracer()
    agents = [
        cls(provider=provider, tracer=tracer)
        for cls in (SymbolicAgent, ToolAgent, DualProcessAgent, AblationAgent)
    ]
    return run_sweep(agents, default_suite(n_perturbed=6, n_novel=3))


def test_sweep_covers_every_agent_and_item(sweep):
    items = default_suite(n_perturbed=6, n_novel=3)  # same args as the fixture
    assert len(sweep.rows) == len(items) * 4
    assert sweep.agents() == ["symbolic", "tool", "dual", "ablation"]


def test_grounding_beats_the_control_on_perturbed_items(sweep):
    """The comparison the whole repo exists to make."""
    grounded = sweep.metrics("tool", "perturbed").accuracy
    control = sweep.metrics("ablation", "perturbed").accuracy
    assert grounded > control


def test_grounded_arm_transfers_off_the_canonical_items(sweep):
    """Both gaps near zero is the only result that supports 'it reasons'."""
    assert abs(sweep.wording_gap("tool")) < 0.2
    assert abs(sweep.number_gap("tool")) < 0.2


def test_control_is_worse_than_grounded_on_every_non_canonical_set(sweep):
    for item_set in ("surface", "perturbed", "novel"):
        assert (
            sweep.metrics("ablation", item_set).accuracy < sweep.metrics("tool", item_set).accuracy
        ), item_set


def test_gap_metrics_are_computed_correctly():
    """Deterministic unit test of the two gaps, independent of any provider."""
    from crt_agent.bench.runner import SweepResult
    from crt_agent.schemas import Answer, Item, Trace

    def row(item_set, correct):
        item = Item(
            item_id=f"{item_set}-{correct}",
            family="bat_ball",
            item_set=item_set,
            text="t",
            answer=1.0,
            lure=2.0,
        )
        answer = Answer(
            agent="a",
            item_id=item.item_id,
            value=1.0 if correct else 2.0,
            trace=Trace(agent="a", item_id=item.item_id),
        )
        return item, answer

    result = SweepResult()
    for item, answer in [
        row("canonical", True),
        row("canonical", True),  # canonical 100%
        row("surface", True),
        row("surface", False),  # surface   50%  -> wording gap 50%
        row("perturbed", False),
        row("perturbed", False),  # perturbed  0%  -> number  gap 100%
    ]:
        result.add(item, answer)

    assert result.wording_gap("a") == pytest.approx(0.5)
    assert result.number_gap("a") == pytest.approx(1.0)
    assert math.isnan(result.wording_gap("nobody"))
    assert result.metrics("a").lure_rate == pytest.approx(0.5)


def test_control_failures_are_lures_not_noise(sweep):
    """A wrong answer that is the designed lure is a different fact than a random miss."""
    m = sweep.metrics("ablation")
    assert m.lure_rate > 0.5 * (1 - m.accuracy)


def test_dual_process_overrides_system_one(sweep):
    m = sweep.metrics("dual")
    assert m.intuitive_n == m.n
    assert m.override_rate > 0.5


def test_audit_validity_is_reported_for_grounded_arms_only(sweep):
    assert sweep.metrics("tool").audit_validity == pytest.approx(1.0)
    assert math.isnan(sweep.metrics("ablation").audit_validity)


def test_metrics_handle_an_empty_scope():
    from crt_agent.bench.runner import SweepResult

    m = SweepResult().metrics("nobody")
    assert m.n == 0 and m.accuracy == 0.0 and math.isnan(m.override_rate)


def test_reports_render(sweep):
    table = summary_table(sweep)
    assert "wording" in table and "number" in table and "tool" in table
    assert "MISSES" in failure_digest(sweep, limit=2)
    item, answer = sweep.rows[0]
    rendered = render_answer(item, answer)
    assert "REASONING" in rendered and "AUDIT" in rendered


def test_persistence_round_trip(tmp_path, sweep):
    repo = Repository(url=f"sqlite:///{tmp_path / 't.db'}")
    repo.create_schema()
    run_id = repo.save_run(
        provider="mock", model="mock", seed=1, agents=sweep.agents(), results=sweep.rows
    )
    assert repo.latest_run_id() == run_id
    attempts = repo.attempts(run_id)
    assert len(attempts) == len(sweep.rows)
    assert any(a.correct for a in attempts)
    with repo.session() as s:
        from crt_agent.store.models import TraceStepRow

        assert s.query(TraceStepRow).count() > len(attempts)


def test_cli_bench_runs_end_to_end(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["bench", "--perturbed", "2", "--novel", "1", "--failures", "2"]) == 0
    out = capsys.readouterr().out
    assert "ACCURACY BY ITEM SET" in out and "ablation" in out


def test_cli_ask_prints_a_trace(capsys):
    question = (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost, in dollars?"
    )
    assert main(["ask", question, "--agent", "tool"]) == 0
    out = capsys.readouterr().out
    assert "PARSE" in out and "SOLVE" in out and "0.05" in out


def test_cli_agents_and_items(capsys):
    assert main(["agents"]) == 0
    assert main(["items", "--set", "canonical"]) == 0
    assert "trap:" in capsys.readouterr().out


def test_ad_hoc_questions_are_not_scored_against_a_missing_answer(provider, tracer):
    """`crt ask` has no reference answer; printing "[wrong]" for a good answer is a lie."""
    from crt_agent.agents import ToolAgent
    from crt_agent.cli import _ad_hoc_item

    item = _ad_hoc_item(
        "A kettle and a mug cost $3.40 in total. The kettle costs $3.00 more than "
        "the mug. How much does the mug cost, in dollars?"
    )
    answer = ToolAgent(provider=provider, tracer=tracer).answer(item)
    rendered = render_answer(item, answer)

    assert answer.value == pytest.approx(0.20)
    assert answer.unit == "dollars", "unit should come from the spec when the item has none"
    assert "unscored" in rendered
    assert "EXPECTED" not in rendered
    assert "nan" not in rendered.lower()
