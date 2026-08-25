"""Arm 4 — the control. Same model, same items, no solver, no schema constraint.

    direct (LLM) -> END

This arm exists to make every other number mean something. Without it, "the tool
agent scored 94%" is unfalsifiable as a claim about the *scaffolding*: maybe the
model would have scored 94% on its own.

It is also the arm where the memorisation effect is visible. Expect it to do well on
the canonical wordings and worse on perturbed ones. That delta — not its absolute
accuracy — is the measurement.

Its traces are always marked `applicable=False` by the auditor, because there is no
derivation to audit. That is the honest reporting, not a bug.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from crt_agent.agents.base import Agent, AgentState
from crt_agent.llm.client import DIRECT_TOOL
from crt_agent.schemas import StepKind

DIRECT_SYSTEM = """Answer the question with a single number. Think it through first."""


def direct_node(provider: Any) -> Any:
    def run(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        response = provider.call(
            system=DIRECT_SYSTEM,
            prompt=state["question"],
            tool=DIRECT_TOOL,
        )
        state["value"] = float(response.arguments["value"])
        state["tokens_in"] = state.get("tokens_in", 0) + response.tokens_in
        state["tokens_out"] = state.get("tokens_out", 0) + response.tokens_out
        state["trace"].add(
            StepKind.DIRECT,
            "direct",
            rationale=str(response.arguments.get("reasoning", ""))[:400],
            payload={"value": state["value"]},
            started=t0,
        )
        return state

    return run


class AblationAgent(Agent):
    name = "ablation"
    grounded = False
    blurb = "Control arm: the same model answering directly, with no solver and no schema."

    def build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("direct", direct_node(self.provider))
        g.add_edge(START, "direct")
        g.add_edge("direct", END)
        return g
