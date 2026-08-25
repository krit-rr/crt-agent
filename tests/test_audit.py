"""The auditor's job is to catch answers that the trace does not derive."""

from __future__ import annotations

from crt_agent.schemas import ProblemSpec, StepKind, Trace, Variable
from crt_agent.tracing.audit import audit, spec_fingerprint

SPEC = ProblemSpec(
    variables=[Variable(name="x", description="x")],
    equations=["x = 2 + 3"],
    query="x",
)
OTHER = ProblemSpec(
    variables=[Variable(name="x", description="x")],
    equations=["x = 9 + 9"],
    query="x",
)


def _trace(spec=SPEC, solve_spec_obj=SPEC, value=5.0, verified=True) -> Trace:
    t = Trace(agent="test", item_id="i")
    t.add(StepKind.PARSE, "parse", payload={"spec_fingerprint": spec_fingerprint(spec)})
    t.add(
        StepKind.SOLVE,
        "solve",
        payload={"value": value, "spec_fingerprint": spec_fingerprint(solve_spec_obj)},
    )
    t.add(StepKind.VERIFY, "verify", payload={"verified": verified})
    return t


def test_coherent_trace_is_valid():
    assert audit(_trace(), 5.0).valid


def test_fingerprint_is_stable_under_whitespace_and_ordering():
    a = ProblemSpec(
        variables=[Variable(name="x", description="x"), Variable(name="y", description="y")],
        equations=["x + y = 4", "x - y = 2"],
        query="x",
    )
    b = ProblemSpec(
        variables=[Variable(name="y", description="y"), Variable(name="x", description="x")],
        equations=["x-y = 2", "x  +  y = 4"],
        query="x",
    )
    assert spec_fingerprint(a) == spec_fingerprint(b)


def test_answer_that_does_not_match_the_solver_is_invalid():
    verdict = audit(_trace(), 42.0)
    assert not verdict.valid and "does not match" in verdict.reason


def test_solving_a_different_spec_than_was_parsed_is_invalid():
    """The load-bearing check: formalise one problem, solve another, report a number."""
    verdict = audit(_trace(spec=SPEC, solve_spec_obj=OTHER), 5.0)
    assert not verdict.valid and "different spec" in verdict.reason


def test_missing_parse_step_is_invalid():
    t = Trace(agent="test", item_id="i")
    t.add(StepKind.SOLVE, "solve", payload={"value": 5.0})
    assert not audit(t, 5.0).valid


def test_failed_verification_is_invalid():
    assert not audit(_trace(verified=False), 5.0).valid


def test_abstention_with_no_value_is_valid():
    t = Trace(agent="test", item_id="i")
    t.add(StepKind.ABSTAIN, "parse", "could not formalise")
    assert audit(t, None).valid


def test_control_arm_is_reported_as_not_applicable_rather_than_failing():
    verdict = audit(_trace(), 5.0, grounded=False)
    assert not verdict.applicable and not verdict.valid
