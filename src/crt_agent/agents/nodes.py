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

GROUNDING THE NUMBERS  (this is where formalisations usually fail)
- Every number stated in the problem must appear as a literal in your equations.
  If the prose says "12 per day", write `12`, not an unbound variable `seen_per_day`.
  Naming a quantity you never bind to a value leaves the system under-determined and
  the solver will reject it.
- The system must pin down the queried variable to a single number. Before you
  finish, check: could a solver derive one value for it from these equations alone?
- If the problem is about proportions of a whole that is never given a size, fix the
  whole at 1. "covers the entire lake" becomes `= 1`, not `= total_area`.

An external symbolic solver computes the answer from your equations. If your model is
wrong, the answer will be wrong, and that is the signal we want."""

REPAIR_SYSTEM = (
    FORMALISE_SYSTEM
    + """

You are being shown a formalisation you produced that the solver rejected, and the
error it gave. Produce a corrected formalisation of the SAME problem.

The error is almost always one of:
- a declared variable that never gets a numeric value (bind it to the literal from
  the prose, or drop it),
- an unknown that no equation constrains (add the missing relationship),
- a quantity treated as symbolic when the problem states it outright.

Do not change what the problem is asking. Fix the algebra."""
)

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
        reliable = bool(getattr(provider, "supports_unreflective_sampling", True))
        state["intuitive_value"] = value
        state["intuitive_reliable"] = reliable
        state["tokens_in"] = state.get("tokens_in", 0) + response.tokens_in
        state["tokens_out"] = state.get("tokens_out", 0) + response.tokens_out
        state["cost_usd"] = state.get("cost_usd", 0.0) + response.cost_usd
        note = str(response.arguments.get("gut_feel", ""))[:200]
        if not reliable:
            note = (
                f"{note}  [NOT a System 1 sample: {provider.name} deliberates before "
                "replying, so this is a second considered answer]"
            ).strip()
        state["trace"].add(
            StepKind.INTUIT,
            "intuit",
            rationale=note,
            payload={"value": value, "unreflective": reliable},
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
        state["cost_usd"] = state.get("cost_usd", 0.0) + response.cost_usd

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


def repair_node(provider: LLMProvider) -> Any:
    """Re-formalise after the solver rejected the spec.

    This is the only place in the system where a model gets to see feedback and try
    again, and it is deliberately narrow: it sees its own equations and the solver's
    complaint, nothing else. It never sees the expected answer, so it cannot converge
    on a number by trial and error — only on a *well-formed system*.

    That distinction is what keeps the repair loop honest. A loop that retried until
    the answer matched would be fitting to the label; this one retries until the
    algebra is solvable, and the algebra is then right or wrong on its own merits.

    Observed failure it exists to fix: models routinely declare `seen_per_day` as a
    symbol and never bind it to the 12 stated in the prose, leaving an under-determined
    system. The solver catches it; this gives the model one chance to hear that.
    """

    def run(state: AgentState) -> AgentState:
        if state.get("repairs", 0) >= state.get("max_repairs", 0):
            return state

        t0 = time.perf_counter()
        state["repairs"] = state.get("repairs", 0) + 1
        previous = state.get("spec")
        complaint = state.get("error") or "the solver rejected the system"

        prompt = (
            f"PROBLEM\n{state['question']}\n\n"
            f"YOUR FORMALISATION\n"
            f"variables: {[v.name for v in previous.variables] if previous else []}\n"
            f"equations: {previous.equations if previous else []}\n"
            f"query: {previous.query if previous else '?'}\n\n"
            f"SOLVER ERROR\n{complaint}"
        )

        response = provider.call(system=REPAIR_SYSTEM, prompt=prompt, tool=FORMALISE_TOOL)
        state["tokens_in"] = state.get("tokens_in", 0) + response.tokens_in
        state["tokens_out"] = state.get("tokens_out", 0) + response.tokens_out
        state["cost_usd"] = state.get("cost_usd", 0.0) + response.cost_usd

        try:
            spec = response.as_spec()
        except ValidationError as exc:
            state["trace"].add(
                StepKind.REPAIR,
                "repair",
                rationale=f"repair produced an invalid spec: {exc.error_count()} violation(s)",
                started=t0,
            )
            return state  # stays abstained

        # Clear the failure so solve_node will run again.
        state["spec"] = spec
        state["abstained"] = False
        state["error"] = None
        state["solver"] = None
        state["trace"].add(
            StepKind.REPAIR,
            "repair",
            rationale=f"re-formalised after: {complaint}",
            payload={"attempt": state["repairs"], "previous_error": complaint},
            started=t0,
        )
        state["trace"].add(
            StepKind.PARSE,
            "repair",
            rationale="; ".join(spec.equations) + f"  ->  solve for {spec.query}",
            payload={
                "spec": spec.model_dump(),
                "spec_fingerprint": spec_fingerprint(spec),
                "repaired": True,
            },
        )
        return state

    return run
