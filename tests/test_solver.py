"""The solver is the trust anchor. It gets the most adversarial tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crt_agent.items import default_suite
from crt_agent.schemas import ProblemSpec, Variable
from crt_agent.solver.symbolic import SolverError, solve_spec


def spec(equations, query="x", names=("x",)):
    return ProblemSpec(
        variables=[Variable(name=n, description=n) for n in names],
        equations=equations,
        query=query,
    )


def test_bat_and_ball_is_exact_not_floating_point():
    result = solve_spec(
        spec(["big + small = 1.10", "big - small = 1.00"], query="small", names=("big", "small"))
    )
    assert result.exact == "1/20"  # not 0.050000000000000044
    assert result.value == pytest.approx(0.05)
    assert result.verified


def test_every_reference_spec_solves_to_its_own_answer():
    for item in default_suite(n_perturbed=10, n_novel=4):
        result = solve_spec(item.reference_spec)
        assert item.is_correct(result.value), f"{item.item_id}: {result.value} != {item.answer}"
        assert result.verified


def test_exponential_growth_is_handled():
    assert solve_spec(spec(["2 ^ x = 2 ^ 48 / 2"])).value == pytest.approx(47)


def test_underdetermined_system_is_an_error_not_a_guess():
    with pytest.raises(SolverError, match="does not determine|under-determined"):
        solve_spec(spec(["x + y = 10"], query="x", names=("x", "y")))


def test_inconsistent_system_is_an_error():
    with pytest.raises(SolverError, match="no solution"):
        solve_spec(spec(["x = 1", "x = 2"]))


def test_undeclared_symbols_are_rejected():
    with pytest.raises(SolverError, match="undeclared"):
        solve_spec(spec(["x = y + 1"]))


@pytest.mark.parametrize(
    "payload",
    [
        '__import__("os").system("echo pwned")',
        "eval('1+1')",
        "open('/etc/passwd')",
    ],
)
def test_code_injection_never_reaches_the_evaluator(payload):
    """Charset validation rejects these at the schema boundary, before sympy sees them."""
    with pytest.raises(ValidationError):
        spec([f"x = {payload}"])


def test_sympy_functions_are_not_in_scope():
    """Even a charset-legal call cannot resolve: the eval namespace is sealed."""
    with pytest.raises(SolverError):
        solve_spec(spec(["x = exp(1)"]))


def test_equation_must_have_exactly_one_equals():
    with pytest.raises(ValidationError):
        spec(["x = 1 = 2"])
    with pytest.raises(ValidationError):
        spec(["x + 1"])


def test_query_must_be_a_declared_variable():
    with pytest.raises(ValidationError):
        spec(["x = 1"], query="y")
