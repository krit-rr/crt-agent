"""Arm 1 — pure symbolic. No model anywhere in the loop.

    match -> solve -> verify

Perfectly transparent, perfectly reproducible, and completely unable to handle a
sentence nobody anticipated. It abstains rather than guessing, which makes its
abstention rate the cleanest available measure of how much linguistic variation the
item set actually contains.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from crt_agent.agents.base import Agent, AgentState
from crt_agent.agents.matcher import match_spec
from crt_agent.agents.nodes import solve_node, verify_node
from crt_agent.schemas import StepKind
from crt_agent.tracing.audit import spec_fingerprint


def match_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    matched = match_spec(state["question"])
    if matched is None:
        state["abstained"] = True
        state["error"] = "no rule matched this phrasing"
        state["trace"].add(StepKind.ABSTAIN, "match", rationale=state["error"], started=t0)
        return state

    family, spec = matched
    state["spec"] = spec
    state["trace"].add(
        StepKind.MATCH,
        "match",
        rationale=f"matched the {family} rule",
        payload={"family": family},
        started=t0,
    )
    state["trace"].add(
        StepKind.PARSE,
        "match",
        rationale="; ".join(spec.equations) + f"  ->  solve for {spec.query}",
        payload={"spec": spec.model_dump(), "spec_fingerprint": spec_fingerprint(spec)},
    )
    return state


def _route(state: AgentState) -> str:
    return END if state.get("abstained") else "solve"


class SymbolicAgent(Agent):
    name = "symbolic"
    grounded = True
    blurb = "Regex templates -> algebra -> sympy. No LLM. Abstains on unseen phrasings."

    def build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("match", match_node)
        g.add_node("solve", solve_node)
        g.add_node("verify", verify_node)
        g.add_edge(START, "match")
        g.add_conditional_edges("match", _route, {"solve": "solve", END: END})
        g.add_edge("solve", "verify")
        g.add_edge("verify", END)
        return g
