"""Cross-model sweeps, the spend ceiling, concurrency, and the System 1 gate.

These cover the machinery added for running the benchmark without an API key. The
System 1 gate is the one worth reading: a benchmark that prints a plausible number for
something it did not actually measure is worse than one that prints nothing.
"""

from __future__ import annotations

import math
import threading

import pytest

from crt_agent.agents import AblationAgent, DualProcessAgent, SymbolicAgent, ToolAgent
from crt_agent.bench import matrix_table, run_matrix, run_sweep, summary_table
from crt_agent.items import canonical_items, default_suite
from crt_agent.llm.client import LLMResponse
from crt_agent.llm.mock import MockProvider
from crt_agent.tracing.langfuse_client import NullTracer


class CostedProvider(MockProvider):
    """Mock that reports spend, so budget logic has something to count."""

    def __init__(self, cost_per_call: float = 0.10, unreflective: bool = True):
        self.cost_per_call = cost_per_call
        self.supports_unreflective_sampling = unreflective
        self.calls = 0
        self._lock = threading.Lock()

    def call(self, **kwargs) -> LLMResponse:
        with self._lock:
            self.calls += 1
        response = super().call(**kwargs)
        response.cost_usd = self.cost_per_call
        return response


def build(cls, provider):
    return cls(provider=provider, tracer=NullTracer())


# -- System 1 gating ------------------------------------------------------------------


def test_system1_metric_is_withheld_when_the_backend_always_deliberates():
    """`claude -p` reasons before replying, so its 'snap judgement' isn't one."""
    provider = CostedProvider(unreflective=False)
    sweep = run_sweep([build(DualProcessAgent, provider)], canonical_items())

    metrics = sweep.metrics("dual")
    assert metrics.intuitive_n == 3, "the step still runs and is still recorded"
    assert metrics.intuitive_reliable_n == 0, "but none of it counts as System 1"
    assert math.isnan(metrics.system1_lure_rate), "so the metric refuses to produce a number"
    assert "  -  " in summary_table(sweep), "and the table prints a dash"


def test_system1_metric_is_reported_when_the_backend_can_sample_unreflectively():
    provider = CostedProvider(unreflective=True)
    sweep = run_sweep([build(DualProcessAgent, provider)], canonical_items())
    metrics = sweep.metrics("dual")
    assert metrics.intuitive_reliable_n == 3
    assert metrics.system1_lure_rate == pytest.approx(1.0)


def test_the_unreliable_intuition_is_flagged_in_the_trace_itself():
    provider = CostedProvider(unreflective=False)
    answer = build(DualProcessAgent, provider).answer(canonical_items()[0])
    from crt_agent.schemas import StepKind

    step = answer.trace.last(StepKind.INTUIT)
    assert step.payload["unreflective"] is False
    assert "NOT a System 1 sample" in step.rationale
    assert answer.intuitive_reliable is False


# -- budget ceiling -------------------------------------------------------------------


def test_sweep_stops_when_the_ceiling_is_reached():
    provider = CostedProvider(cost_per_call=1.0)
    items = default_suite(n_perturbed=8, n_novel=4)
    sweep = run_sweep([build(ToolAgent, provider)], items, max_cost_usd=5.0)

    spent = sweep.metrics("tool").cost_usd
    assert spent >= 5.0, "the ceiling is reached, not merely approached"
    assert len(sweep.rows) < len(items), "and the sweep stopped short"


def test_no_ceiling_means_no_early_stop():
    provider = CostedProvider(cost_per_call=1.0)
    items = default_suite(n_perturbed=2, n_novel=1)
    sweep = run_sweep([build(ToolAgent, provider)], items)
    assert len(sweep.rows) == len(items)


def test_partial_results_are_still_scoreable():
    """A truncated run must degrade to a smaller n, not to an exception."""
    provider = CostedProvider(cost_per_call=1.0)
    items = default_suite(n_perturbed=8, n_novel=4)
    sweep = run_sweep([build(ToolAgent, provider)], items, max_cost_usd=4.0)
    table = summary_table(sweep)
    assert "ACCURACY BY ITEM SET" in table
    assert sweep.metrics("tool").n > 0


def test_free_backends_are_unaffected_by_a_ceiling():
    """A provider reporting zero cost must never trip the guard."""
    items = default_suite(n_perturbed=3, n_novel=2)
    sweep = run_sweep([build(ToolAgent, MockProvider())], items, max_cost_usd=0.01)
    assert len(sweep.rows) == len(items)


# -- concurrency ----------------------------------------------------------------------


def test_concurrent_and_serial_sweeps_agree():
    items = default_suite(n_perturbed=4, n_novel=2)
    serial = run_sweep([build(ToolAgent, MockProvider())], items, concurrency=1)
    parallel = run_sweep([build(ToolAgent, MockProvider())], items, concurrency=6)

    assert len(serial.rows) == len(parallel.rows)
    assert serial.metrics("tool").accuracy == parallel.metrics("tool").accuracy

    # Order differs under concurrency, so compare as a set of outcomes.
    def outcomes(sweep):
        return sorted((i.item_id, a.value) for i, a in sweep.rows)

    assert outcomes(serial) == outcomes(parallel)


def test_every_item_is_attempted_exactly_once_under_concurrency():
    items = default_suite(n_perturbed=5, n_novel=3)
    sweep = run_sweep([build(ToolAgent, MockProvider())], items, concurrency=8)
    seen = [i.item_id for i, _ in sweep.rows]
    assert sorted(seen) == sorted(i.item_id for i in items)
    assert len(seen) == len(set(seen))


# -- cross-model matrix ---------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix():
    items = default_suite(n_perturbed=4, n_novel=2)

    def build_agents(model: str):
        provider = CostedProvider(cost_per_call=0.01)
        provider.name = f"fake:{model}"
        return [
            build(cls, provider)
            for cls in (SymbolicAgent, ToolAgent, DualProcessAgent, AblationAgent)
        ]

    return run_matrix(["fast-model", "slow-model"], build_agents, items)


def test_matrix_runs_every_model(matrix):
    assert matrix.models == ["fast-model", "slow-model"]
    assert matrix.agents() == ["symbolic", "tool", "dual", "ablation"]


def test_grounding_delta_is_the_headline_number(matrix):
    for model in matrix.models:
        delta = matrix.grounding_delta(model)
        expected = (
            matrix.metrics(model, "tool").accuracy - matrix.metrics(model, "ablation").accuracy
        )
        assert delta == pytest.approx(expected)
        assert delta > 0, "grounding should beat the control on the mock"


def test_grounding_delta_is_nan_without_a_control_arm():
    items = canonical_items()
    solo = run_matrix(["m"], lambda _: [build(ToolAgent, MockProvider())], items)
    assert math.isnan(solo.grounding_delta("m"))


def test_matrix_table_renders_and_totals_cost(matrix):
    table = matrix_table(matrix)
    assert "MODELS x ARMS" in table
    assert "grounding" in table
    for model in matrix.models:
        assert model in table
    assert matrix.total_cost() > 0
    assert f"${matrix.total_cost():.2f}" in table


def test_matrix_budget_drops_later_models_rather_than_half_measuring_all():
    items = default_suite(n_perturbed=6, n_novel=3)

    def build_agents(model: str):
        return [build(ToolAgent, CostedProvider(cost_per_call=1.0))]

    result = run_matrix(["a", "b", "c"], build_agents, items, max_cost_usd=5.0)
    assert "a" in result.models
    assert "c" not in result.models, "the ceiling should cut the tail, not thin every model"


def test_override_rate_is_gated_on_the_same_condition_as_system1_lure():
    """Two deliberate answers differing is not System 2 overriding System 1."""
    unreflective = run_sweep(
        [build(DualProcessAgent, CostedProvider(unreflective=False))], canonical_items()
    )
    assert math.isnan(unreflective.metrics("dual").override_rate)

    reflective = run_sweep(
        [build(DualProcessAgent, CostedProvider(unreflective=True))], canonical_items()
    )
    assert not math.isnan(reflective.metrics("dual").override_rate)
