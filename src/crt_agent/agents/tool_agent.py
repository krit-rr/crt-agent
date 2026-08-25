"""Arm 2 — LLM formalises, solver computes.

    parse (LLM) -> solve (sympy) -> verify

The model reads prose and emits `ProblemSpec`; it has no channel through which to
state a number. Every arithmetic step is done by code that never saw the sentence.

This is the arm that answers the original question — "can I see it reason rather than
recall?" — because the trace contains the equations it committed to, and the answer
provably comes from those equations and nothing else.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from crt_agent.agents.base import Agent, AgentState
from crt_agent.agents.nodes import parse_node, repair_node, solve_node, verify_node


def _after_parse(state: AgentState) -> str:
    return END if state.get("abstained") else "solve"


def _after_solve(state: AgentState) -> str:
    """The one cycle in the graph: a rejected spec gets one more shot.

    Only solver *rejections* route to repair. A spec that solved cleanly goes
    straight to verify, and a spec that has already used its repair budget goes to
    verify too (where the abstention is recorded honestly).
    """
    if state.get("abstained") and state.get("repairs", 0) < state.get("max_repairs", 0):
        return "repair"
    return "verify"


def _after_repair(state: AgentState) -> str:
    return END if state.get("abstained") else "solve"


class ToolAgent(Agent):
    name = "tool"
    grounded = True
    blurb = "LLM emits equations under a forced tool schema; sympy solves them."
    max_repairs = 1

    def build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("parse", parse_node(self.provider))
        g.add_node("solve", solve_node)
        g.add_node("repair", repair_node(self.provider))
        g.add_node("verify", verify_node)
        g.add_edge(START, "parse")
        g.add_conditional_edges("parse", _after_parse, {"solve": "solve", END: END})
        g.add_conditional_edges("solve", _after_solve, {"repair": "repair", "verify": "verify"})
        g.add_conditional_edges("repair", _after_repair, {"solve": "solve", END: END})
        g.add_edge("verify", END)
        return g
