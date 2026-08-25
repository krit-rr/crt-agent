"""Agent behaviour, all against the deterministic mock provider."""

from __future__ import annotations

import pytest

from crt_agent.agents import AGENTS, AblationAgent, DualProcessAgent, SymbolicAgent, ToolAgent
from crt_agent.items import canonical_items, default_suite
from crt_agent.schemas import Item, StepKind


@pytest.fixture(scope="module")
def canonical():
    return {i.family: i for i in canonical_items()}


@pytest.mark.parametrize("cls", list(AGENTS.values()), ids=list(AGENTS))
def test_every_agent_answers_every_item_without_raising(cls, provider, tracer):
    agent = cls(provider=provider, tracer=tracer)
    for item in default_suite(3, 2):
        answer = agent.answer(item)
        assert answer.agent == cls.name
        assert answer.trace.steps, "an agent must always leave a trace"


@pytest.mark.parametrize("cls", [SymbolicAgent, ToolAgent, DualProcessAgent])
def test_grounded_agents_get_the_bat_and_ball_right(cls, provider, tracer, canonical):
    answer = cls(provider=provider, tracer=tracer).answer(canonical["bat_ball"])
    assert answer.value == pytest.approx(0.05)
    assert answer.audit.valid


@pytest.mark.parametrize("cls", [SymbolicAgent, ToolAgent, DualProcessAgent])
def test_grounded_agents_never_report_an_unaudited_answer(cls, provider, tracer):
    agent = cls(provider=provider, tracer=tracer)
    for item in default_suite(4, 2):
        answer = agent.answer(item)
        assert answer.audit.applicable
        assert answer.audit.valid, f"{item.item_id}: {answer.audit.reason}"


def test_symbolic_agent_abstains_on_an_unknown_phrasing(provider, tracer):
    item = Item(
        item_id="x",
        family="bat_ball",
        item_set="novel",
        answer=1,
        lure=2,
        text="Two trains leave stations 300 km apart. When do they meet?",
    )
    answer = SymbolicAgent(provider=provider, tracer=tracer).answer(item)
    assert answer.abstained and answer.value is None
    assert answer.trace.last(StepKind.ABSTAIN) is not None
    assert answer.audit.valid, "a clean abstention is an honest outcome"


def test_tool_agent_trace_records_the_equations_it_committed_to(provider, tracer, canonical):
    answer = ToolAgent(provider=provider, tracer=tracer).answer(canonical["widgets"])
    parse = answer.trace.last(StepKind.PARSE)
    assert parse is not None
    equations = parse.payload["spec"]["equations"]
    assert any("rate" in e for e in equations)
    assert not any(e.strip().startswith("minutes = 5") for e in equations), "must not pre-solve"


def test_dual_process_samples_system1_before_it_formalises(provider, tracer, canonical):
    answer = DualProcessAgent(provider=provider, tracer=tracer).answer(canonical["lily_pad"])
    kinds = [s.kind for s in answer.trace.steps]
    assert kinds.index(StepKind.INTUIT) < kinds.index(StepKind.PARSE)


def test_dual_process_reports_the_override(provider, tracer, canonical):
    item = canonical["bat_ball"]
    answer = DualProcessAgent(provider=provider, tracer=tracer).answer(item)
    assert item.is_lure(answer.intuitive_value)
    assert item.is_correct(answer.value)
    assert answer.conflict_detected


def test_ablation_is_marked_unauditable_by_design(provider, tracer, canonical):
    answer = AblationAgent(provider=provider, tracer=tracer).answer(canonical["bat_ball"])
    assert answer.audit is not None
    assert not answer.audit.applicable
    assert answer.trace.last(StepKind.SOLVE) is None


def test_agent_survives_a_provider_that_explodes(tracer, canonical):
    class Broken:
        name = "broken"

        def call(self, **_):
            raise RuntimeError("upstream 500")

    answer = ToolAgent(provider=Broken(), tracer=tracer).answer(canonical["bat_ball"])
    assert answer.value is None
    assert "upstream 500" in (answer.error or "")
