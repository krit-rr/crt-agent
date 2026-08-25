"""Shared agent scaffolding.

Every agent is a `StateGraph` over the same state type, so the four architectures
differ only in which nodes exist and how they're wired — which is the comparison the
benchmark is trying to make. Anything shared (timing, tracing, audit, answer
assembly) lives here so it can't drift between arms.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from crt_agent.llm.client import LLMProvider, build_provider
from crt_agent.schemas import Answer, Item, StepKind, Trace
from crt_agent.tracing.audit import audit
from crt_agent.tracing.langfuse_client import Tracer, get_tracer


class AgentState(TypedDict, total=False):
    """The value that flows through every graph.

    `trace` is mutated in place by the nodes. LangGraph is happy with that because
    we never fan out in parallel; if these graphs ever gain concurrent branches the
    trace needs a reducer instead.
    """

    item_id: str
    question: str
    trace: Trace
    spec: Any  # ProblemSpec | None
    solver: Any  # SolverResult | None
    value: float | None
    intuitive_value: float | None
    conflict: bool
    abstained: bool
    error: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    intuitive_reliable: bool
    repairs: int
    max_repairs: int


class Agent(ABC):
    """Base class: builds the graph once, then runs items through it."""

    #: display name used in reports and the database
    name: str = "agent"
    #: whether this arm has a solver the auditor can check against
    grounded: bool = True
    #: one-line description shown in `crt agents`
    blurb: str = ""
    #: how many times the parser may be re-prompted with the solver's rejection
    max_repairs: int = 0

    def __init__(
        self,
        provider: LLMProvider | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider or build_provider()
        self.tracer = tracer or get_tracer()
        self._graph = self.build_graph().compile()

    @abstractmethod
    def build_graph(self) -> Any:
        """Return an uncompiled `StateGraph`."""

    @property
    def samples_unreflectively(self) -> bool:
        """Whether this agent's backend can produce a genuine System 1 answer."""
        return bool(getattr(self.provider, "supports_unreflective_sampling", True))

    def answer(self, item: Item) -> Answer:
        started = time.perf_counter()
        state: AgentState = {
            "item_id": item.item_id,
            "question": item.text,
            "trace": Trace(agent=self.name, item_id=item.item_id),
            "value": None,
            "intuitive_value": None,
            "conflict": False,
            "abstained": False,
            "error": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "intuitive_reliable": True,
            "repairs": 0,
            "max_repairs": self.max_repairs,
        }

        with self.tracer.span(
            f"crt/{self.name}",
            input={"item_id": item.item_id, "question": item.text},
            metadata={"family": item.family, "item_set": item.item_set},
        ):
            try:
                final: AgentState = self._graph.invoke(state)  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001 - one bad item must not kill a sweep
                state["error"] = f"{type(exc).__name__}: {exc}"
                state["trace"].add(StepKind.ABSTAIN, "graph", f"unhandled error: {exc}")
                final = state

        trace = final["trace"]
        solver = final.get("solver")
        return Answer(
            agent=self.name,
            item_id=item.item_id,
            value=final.get("value"),
            # An ad-hoc question carries no unit; fall back to the one the parser
            # declared in its spec so `crt ask` still reports "0.05 dollars".
            unit=item.unit or (solver.unit if solver else ""),
            abstained=bool(final.get("abstained")),
            error=final.get("error"),
            trace=trace,
            audit=audit(trace, final.get("value"), grounded=self.grounded),
            intuitive_value=final.get("intuitive_value"),
            intuitive_reliable=bool(final.get("intuitive_reliable", True)),
            conflict_detected=bool(final.get("conflict")),
            latency_ms=(time.perf_counter() - started) * 1000,
            tokens_in=int(final.get("tokens_in") or 0),
            tokens_out=int(final.get("tokens_out") or 0),
            cost_usd=float(final.get("cost_usd") or 0.0),
        )
