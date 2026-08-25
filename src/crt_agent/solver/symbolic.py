"""Deterministic algebra over a `ProblemSpec`.

This module contains no model, no heuristics and no knowledge of CRT. It takes
declared symbols and equations, solves the system, substitutes the solution back in,
and refuses to return anything whose residuals are non-zero.

Two properties matter:

* **Sealed input.** Only the declared variable names become symbols. `parse_expr` is
  given an empty global namespace, so nothing else can resolve — a spec containing
  `__import__` or `pi` or a stray function call fails to parse rather than evaluating.
* **Exact arithmetic.** Decimals in the spec are converted to `Rational`, so
  ``(1.10 - 1.00) / 2`` is ``1/20`` and not ``0.050000000000000044``.
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    rationalize,
    standard_transformations,
)

from crt_agent.schemas import ProblemSpec, SolverResult

# `rationalize` turns 1.10 into 11/10 at parse time, so the whole pipeline stays exact.
# `convert_xor` lets a spec write `2 ^ day` for exponentiation.
_TRANSFORMS = standard_transformations + (convert_xor, rationalize)

# The sealed namespace the parsed code is evaluated in. sympy's tokenizer rewrites
# numeric literals into these constructors, so they must be present -- and nothing
# else is. No builtins, no sympy functions, no `__import__`.
_SEALED_GLOBALS = {
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
}


class SolverError(RuntimeError):
    """Raised when a spec is unparseable, inconsistent, or under-determined."""


def _to_expr(text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    try:
        return parse_expr(
            text,
            local_dict=dict(symbols),
            global_dict=dict(_SEALED_GLOBALS),  # sealed: numeric constructors only
            transformations=_TRANSFORMS,
            evaluate=True,
        )
    except Exception as exc:  # noqa: BLE001 - sympy raises a wide variety
        raise SolverError(f"could not parse {text!r}: {exc}") from exc


def solve_spec(spec: ProblemSpec) -> SolverResult:
    """Solve a spec and verify the solution by back-substitution."""
    symbols = {v.name: sp.Symbol(v.name, real=True) for v in spec.variables}

    equations: list[sp.Eq] = []
    for raw in spec.equations:
        lhs_text, rhs_text = raw.split("=")
        lhs, rhs = _to_expr(lhs_text, symbols), _to_expr(rhs_text, symbols)
        free = (lhs - rhs).free_symbols - set(symbols.values())
        if free:
            raise SolverError(f"equation {raw!r} uses undeclared symbols: {sorted(map(str, free))}")
        equations.append(sp.Eq(lhs, rhs))

    solutions = sp.solve(equations, list(symbols.values()), dict=True)
    if not solutions:
        raise SolverError("system has no solution")

    solution = solutions[0]
    target = symbols[spec.query]
    if target not in solution:
        raise SolverError(
            f"system does not determine {spec.query!r}; "
            f"solved for {sorted(str(k) for k in solution)}"
        )

    exact = sp.nsimplify(solution[target])
    if exact.free_symbols:
        raise SolverError(f"{spec.query!r} is under-determined: {exact}")

    residuals: dict[str, str] = {}
    verified = True
    for raw, eq in zip(spec.equations, equations, strict=True):
        residual = sp.simplify((eq.lhs - eq.rhs).subs(solution))
        residuals[raw] = str(residual)
        if residual != 0:
            verified = False

    return SolverResult(
        value=float(exact),
        exact=str(exact),
        unit=spec.unit,
        all_solutions={str(k): float(v) for k, v in solution.items() if not v.free_symbols},
        verified=verified,
        residuals=residuals,
    )
