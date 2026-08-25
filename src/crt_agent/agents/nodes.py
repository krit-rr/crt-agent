"""Reusable graph nodes.

The three grounded agents share `solve_node` and `verify_node` verbatim. That's on
purpose: if the arms differed in how they solved, a difference in scores wouldn't be
attributable to the architecture.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from crt_agent.agents.base import AgentState
from crt_agent.llm.client import FORMALISE_TOOL, INTUIT_TOOL, LLMProvider
from crt_agent.schemas import StepKind
from crt_agent.solver.symbolic import SolverError, solve_spec
from crt_agent.tracing.audit import spec_fingerprint

FORMALISE_SYSTEM = """You translate word problems into algebra. You never compute answers.

Rules:
- Declare every quantity you need, including intermediate ones, in snake_case.
- Write equations using only those names, numbers, and + - * / ( ) ^.
- Exactly one '=' per equation.
- Do not pre-solve. `small = 0.05` is a violation; `big + small = 1.10` is correct.
- If the prose implies a relationship you had to interpret, record it in assumptions.

An external symbolic solver computes the answer from your equations. If your model is
wrong, the answer will be wrong, and that is the signal we want."""

INTUIT_SYSTEM = """Answer with the very first number that comes to mind.

Do not check it. Do not work anything out. Do not reconsider. We are deliberately
sampling the fast, unreflective response - its usefulness depends on it being
genuinely unreflective."""


def intuit_node(provider: LLMProvider) -> Any:
    """System 1 probe: capture the snap judgement before any deliberation happens."""

    def run(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        response = provider.call(
            system=INTUIT_SYSTEM,
            prompt=state["question"],
            tool=INTUIT_TOOL,
            max_tokens=200,
            temperature=1.0,
        )
        value = float(response.arguments["value"])
        state["intuitive_value"] = value
        state["tokens_in"] = state.get("tokens_in", 0) + response.tokens_in
        state["tokens_out"] = state.get("tokens_out", 0) + response.tokens_out
        state["trace"].add(
            StepKind.INTUIT,
            "intuit",
            rationale=str(response.arguments.get("gut_feel", ""))[:200],
            payload={"value": value},
            started=t0,
        )
        return state

    return run


def parse_node(provider: LLMProvider) -> Any:
    """Prose -> `ProblemSpec`, validated before it is allowed downstream."""

    def run(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        response = provider.call(
            system=FORMALISE_SYSTEM,
            prompt=state["question"],
            tool=FORMALISE_TOOL,
        )
        state["tokens_in"] = state.get("tokens_in", 0) + response.tokens_in
        state["tokens_out"] = state.get("tokens_out", 0) + response.tokens_out

        try:
            spec = response.as_spec()
        except ValidationError as exc:
            state["error"] = f"invalid formalisation: {exc.error_count()} schema violation(s)"
            state["abstained"] = True
            state["trace"].add(
                StepKind.ABSTAIN,
                "parse",
                rationale=state["error"],
                payload={"raw": response.arguments},
                started=t0,
            )
            return state

        state["spec"] = spec
        state["trace"].add(
            StepKind.PARSE,
            "parse",
            rationale="; ".join(spec.equations) + f"  ->  solve for {spec.query}",
            payload={
                "spec": spec.model_dump(),
                "spec_fingerprint": spec_fingerprint(spec),
                "assumptions": spec.assumptions,
            },
            started=t0,
        )
        return state

    return run


def solve_node(state: AgentState) -> AgentState:
    """Deterministic. No model involved, and none can be."""
    if state.get("abstained") or state.get("spec") is None:
        return state

    t0 = time.perf_counter()
    spec = state["spec"]
    try:
        result = solve_spec(spec)
    except SolverError as exc:
        state["error"] = f"solver: {exc}"
        state["abstained"] = True
        state["trace"].add(StepKind.ABSTAIN, "solve", rationale=state["error"], started=t0)
        return state

    state["solver"] = result
    state["value"] = result.value
    state["trace"].add(
        StepKind.SOLVE,
        "solve",
        rationale=f"{spec.query} = {result.exact}" + (f" {result.unit}" if result.unit else ""),
        payload={
            "value": result.value,
            "exact": result.exact,
            "spec_fingerprint": spec_fingerprint(spec),
            "all_solutions": result.all_solutions,
        },
        started=t0,
    )
    return state


def verify_node(state: AgentState) -> AgentState:
    """Back-substitute and record the residuals. Cheap, and it catches real bugs."""
    result = state.get("solver")
    if result is None:
        return state

    t0 = time.perf_counter()
    state["trace"].add(
        StepKind.VERIFY,
        "verify",
        rationale=(
            "all residuals zero" if result.verified else f"non-zero residuals: {result.residuals}"
        ),
        payload={"verified": result.verified, "residuals": result.residuals},
        started=t0,
    )
    if not result.verified:
        state["error"] = "verification failed"
        state["abstained"] = True
        state["value"] = None
    return state
