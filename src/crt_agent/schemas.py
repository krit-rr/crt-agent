"""Typed contracts for the whole system.

The single most important idea in this repo lives here: `ProblemSpec`.

An LLM in this system is never allowed to *state an answer*. It is only allowed to
state a **formalisation** — variables, equations, and which variable is being asked
about. The number comes out of a symbolic solver that has never seen the prose.

That constraint is what turns "the model recited $0.05 because the internet is full
of $0.05" into "the model translated the sentence into `bat + ball = 110` and
`bat - ball = 100`, and sympy did the rest."
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------------------
# Problem formalisation
# --------------------------------------------------------------------------------------

# Deliberately narrow. Anything the LLM emits that is not a plain algebraic sentence
# over declared symbols is rejected before it ever reaches the solver.
_ALLOWED_SPEC_CHARS = set("abcdefghijklmnopqrstuvwxyz_0123456789 +-*/()=.,^")


class Variable(BaseModel):
    """One quantity in the problem, named by the parser."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="snake_case identifier used inside equations")
    description: str = Field(description="what this quantity means in the story")
    unit: str = Field(default="", description="e.g. 'cents', 'minutes', 'days'")

    @field_validator("name")
    @classmethod
    def _valid_identifier(cls, v: str) -> str:
        if not v.isidentifier() or v != v.lower():
            raise ValueError(f"variable name must be a lowercase identifier, got {v!r}")
        return v


class ProblemSpec(BaseModel):
    """A word problem, reduced to algebra.

    This is the *only* channel through which an LLM may influence an answer.
    """

    model_config = ConfigDict(frozen=True)

    variables: list[Variable]
    equations: list[str] = Field(
        description="algebraic sentences over the declared variables, e.g. 'bat + ball = 110'"
    )
    query: str = Field(description="name of the variable the question asks for")
    unit: str = Field(default="", description="unit of the answer")
    assumptions: list[str] = Field(
        default_factory=list,
        description="human-readable modelling assumptions the parser committed to",
    )

    @field_validator("equations")
    @classmethod
    def _charset(cls, eqs: list[str]) -> list[str]:
        for eq in eqs:
            bad = set(eq.lower()) - _ALLOWED_SPEC_CHARS
            if bad:
                raise ValueError(f"illegal characters in equation {eq!r}: {sorted(bad)}")
            if eq.count("=") != 1:
                raise ValueError(f"equation must contain exactly one '=': {eq!r}")
        return eqs

    def model_post_init(self, __context: Any) -> None:
        names = {v.name for v in self.variables}
        if self.query not in names:
            raise ValueError(f"query {self.query!r} is not a declared variable ({sorted(names)})")
        if not self.equations:
            raise ValueError("spec must contain at least one equation")


class SolverResult(BaseModel):
    """What the symbolic solver returns. Includes its own verification."""

    model_config = ConfigDict(frozen=True)

    value: float
    exact: str = Field(description="exact rational/symbolic form, e.g. '1/20'")
    unit: str = ""
    all_solutions: dict[str, float] = Field(default_factory=dict)
    verified: bool = Field(
        description="solution was substituted back into every equation and residuals were zero"
    )
    residuals: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Reasoning traces
# --------------------------------------------------------------------------------------


class StepKind(StrEnum):
    """Semantic role of a trace step.

    The auditor cares about these, not about free text. A run is only valid if the
    answer can be reconstructed from steps of the right kinds in the right order.
    """

    INTUIT = "intuit"  # System 1: fast, unaided, deliberately unreflective
    PARSE = "parse"  # prose -> ProblemSpec
    SOLVE = "solve"  # ProblemSpec -> SolverResult (deterministic)
    VERIFY = "verify"  # back-substitution check
    RECONCILE = "reconcile"  # System 1 vs System 2 comparison
    MATCH = "match"  # rule-based template match (symbolic agent)
    DIRECT = "direct"  # ungrounded LLM answer (ablation control only)
    ABSTAIN = "abstain"  # agent declined to answer


class TraceStep(BaseModel):
    """One observable move in the agent's reasoning."""

    index: int
    kind: StepKind
    node: str = Field(description="graph node that produced this step")
    rationale: str = Field(default="", description="human-readable explanation")
    payload: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0

    def render(self) -> str:
        kind = self.kind.value.upper()
        head = f"  [{self.index}] {kind:<9} ({self.node}, {self.duration_ms:.0f}ms)"
        body = f"\n      {self.rationale}" if self.rationale else ""
        return head + body


class Trace(BaseModel):
    """The full reasoning record for one attempt at one item."""

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str
    item_id: str
    steps: list[TraceStep] = Field(default_factory=list)

    def add(
        self,
        kind: StepKind,
        node: str,
        rationale: str = "",
        payload: dict[str, Any] | None = None,
        started: float | None = None,
    ) -> TraceStep:
        step = TraceStep(
            index=len(self.steps),
            kind=kind,
            node=node,
            rationale=rationale,
            payload=payload or {},
            duration_ms=(time.perf_counter() - started) * 1000 if started else 0.0,
        )
        self.steps.append(step)
        return step

    def last(self, kind: StepKind) -> TraceStep | None:
        for step in reversed(self.steps):
            if step.kind is kind:
                return step
        return None

    def render(self) -> str:
        return "\n".join(s.render() for s in self.steps)


class AuditVerdict(BaseModel):
    """Result of checking that a trace actually derives its own answer."""

    valid: bool
    reason: str
    applicable: bool = True  # False for the ungrounded control arm


# --------------------------------------------------------------------------------------
# Items and answers
# --------------------------------------------------------------------------------------


ItemFamily = Literal["bat_ball", "widgets", "lily_pad", "discount", "rank", "drift"]
ItemSet = Literal["canonical", "surface", "perturbed", "novel"]


class Item(BaseModel):
    """One CRT question, plus the two numbers that make it a CRT question.

    `lure` is the wrong answer that System 1 produces. Tracking it separately from
    "wrong" is the whole psychological point: an agent that is wrong in the *lure*
    direction is failing differently from one that is wrong at random.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    family: ItemFamily
    item_set: ItemSet
    text: str
    answer: float
    lure: float
    unit: str = ""
    tolerance: float = 1e-6
    reference_spec: ProblemSpec | None = None

    def is_correct(self, value: float | None) -> bool:
        return value is not None and abs(value - self.answer) <= self.tolerance

    def is_lure(self, value: float | None) -> bool:
        return value is not None and abs(value - self.lure) <= self.tolerance


class Answer(BaseModel):
    """What an agent returns for one item."""

    agent: str
    item_id: str
    value: float | None = None
    unit: str = ""
    abstained: bool = False
    error: str | None = None
    trace: Trace
    audit: AuditVerdict | None = None
    intuitive_value: float | None = Field(
        default=None, description="System 1 answer, when the agent produces one separately"
    )
    conflict_detected: bool = False
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
