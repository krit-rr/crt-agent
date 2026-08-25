"""Arm 3 — dual process (Kahneman's System 1 / System 2, which CRT was built to probe).

    intuit (LLM, fast) ──┐
                         ├─> reconcile
    parse -> solve -> verify ──┘

Frederick's 2005 test isn't a maths test. Every item has an answer that is wrong but
*arrives first*, and the test measures whether you override it. An agent that only
reports its final answer throws away the measurement.

So this arm samples the snap judgement first, at temperature 1.0 and with an explicit
instruction not to check it, and keeps it. `reconcile` then reports whether the two
systems disagreed — which is the closest thing this repo has to a directly observable
act of reflection.

Note the ordering constraint: intuit MUST run before parse. Once the graph has
committed to equations, a "fast" answer sampled afterwards is contaminated by them.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from crt_agent.agents.base import Agent, AgentState
from crt_agent.agents.nodes import (
    intuit_node,
    parse_node,
    repair_node,
    solve_node,
    verify_node,
)
from crt_agent.schemas import StepKind

CONFLICT_TOLERANCE = 1e-6


def reconcile_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    fast, slow = state.get("intuitive_value"), state.get("value")

    if slow is None:
        state["conflict"] = False
        state["trace"].add(
            StepKind.RECONCILE,
            "reconcile",
            rationale="deliberate path produced nothing; the snap judgement is not promoted",
            payload={"intuitive": fast, "deliberate": None},
            started=t0,
        )
        return state

    conflict = fast is not None and abs(fast - slow) > CONFLICT_TOLERANCE
    state["conflict"] = conflict
    state["trace"].add(
        StepKind.RECONCILE,
        "reconcile",
        rationale=(
            f"override: snap answer {fast} rejected in favour of derived {slow}"
            if conflict
            else f"agreement: both systems arrived at {slow}"
        ),
        payload={"intuitive": fast, "deliberate": slow, "conflict": conflict},
        started=t0,
    )
    return state


def _after_parse(state: AgentState) -> str:
    return "reconcile" if state.get("abstained") else "solve"


def _after_solve(state: AgentState) -> str:
    if state.get("abstained") and state.get("repairs", 0) < state.get("max_repairs", 0):
        return "repair"
    return "verify"


def _after_repair(state: AgentState) -> str:
    return "reconcile" if state.get("abstained") else "solve"


class DualProcessAgent(Agent):
    name = "dual"
    grounded = True
    blurb = "Samples a System 1 snap answer, derives a System 2 answer, reports the override."
    max_repairs = 1

    def build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("intuit", intuit_node(self.provider))
        g.add_node("parse", parse_node(self.provider))
        g.add_node("solve", solve_node)
        g.add_node("repair", repair_node(self.provider))
        g.add_node("verify", verify_node)
        g.add_node("reconcile", reconcile_node)
        g.add_edge(START, "intuit")
        g.add_edge("intuit", "parse")
        g.add_conditional_edges("parse", _after_parse, {"solve": "solve", "reconcile": "reconcile"})
        g.add_conditional_edges("solve", _after_solve, {"repair": "repair", "verify": "verify"})
        g.add_conditional_edges(
            "repair", _after_repair, {"solve": "solve", "reconcile": "reconcile"}
        )
        g.add_edge("verify", "reconcile")
        g.add_edge("reconcile", END)
        return g
