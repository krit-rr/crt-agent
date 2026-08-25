"""Benchmark sweep + metrics.

The metrics here are chosen so that a table of results answers the question the
project actually asks. Accuracy alone can't:

    accuracy          did it get the number right
    lure_rate         when wrong, was it wrong in the direction the item was designed to pull
    abstention_rate   did it decline instead of guessing
    audit_validity    is the answer reconstructable from the trace
    override_rate     (dual only) how often System 2 overruled System 1
    wording_gap       accuracy(canonical) - accuracy(surface)
    number_gap        accuracy(canonical) - accuracy(perturbed)

The two gaps are the headline, and they are separate on purpose:

    wording_gap high  ->  it was keyed to the famous phrasing
    number_gap high   ->  it was reciting the famous answer
    both near zero    ->  whatever it is doing, it transfers

A single "recall gap" that perturbs wording and numbers together cannot tell those
apart, which is the first thing anyone will ask you about the result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from crt_agent.agents.base import Agent
from crt_agent.schemas import Answer, Item, ItemSet, StepKind


@dataclass
class Metrics:
    agent: str
    scope: str
    n: int = 0
    correct: int = 0
    lure: int = 0
    abstained: int = 0
    errored: int = 0
    audit_checked: int = 0
    audit_valid: int = 0
    overrides: int = 0
    intuitive_lure: int = 0
    intuitive_n: int = 0
    latency_ms: float = 0.0
    tokens_out: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def lure_rate(self) -> float:
        return self.lure / self.n if self.n else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.abstained / self.n if self.n else 0.0

    @property
    def answered_accuracy(self) -> float:
        """Accuracy over items the agent was willing to answer."""
        answered = self.n - self.abstained
        return self.correct / answered if answered else 0.0

    @property
    def audit_validity(self) -> float:
        return self.audit_valid / self.audit_checked if self.audit_checked else float("nan")

    @property
    def override_rate(self) -> float:
        return self.overrides / self.intuitive_n if self.intuitive_n else float("nan")

    @property
    def system1_lure_rate(self) -> float:
        return self.intuitive_lure / self.intuitive_n if self.intuitive_n else float("nan")

    @property
    def mean_latency_ms(self) -> float:
        return self.latency_ms / self.n if self.n else 0.0


@dataclass
class SweepResult:
    rows: list[tuple[Item, Answer]] = field(default_factory=list)

    def add(self, item: Item, answer: Answer) -> None:
        self.rows.append((item, answer))

    def metrics(self, agent: str, item_set: ItemSet | None = None) -> Metrics:
        scope = item_set or "all"
        m = Metrics(agent=agent, scope=scope)
        for item, answer in self.rows:
            if answer.agent != agent:
                continue
            if item_set and item.item_set != item_set:
                continue
            m.n += 1
            m.correct += item.is_correct(answer.value)
            m.lure += item.is_lure(answer.value)
            m.abstained += answer.abstained
            m.errored += answer.error is not None
            m.latency_ms += answer.latency_ms
            m.tokens_out += answer.tokens_out
            if answer.audit and answer.audit.applicable:
                m.audit_checked += 1
                m.audit_valid += answer.audit.valid
            if answer.intuitive_value is not None:
                m.intuitive_n += 1
                m.intuitive_lure += item.is_lure(answer.intuitive_value)
                m.overrides += answer.conflict_detected
        return m

    def _gap(self, agent: str, against: ItemSet) -> float:
        canonical = self.metrics(agent, "canonical")
        other = self.metrics(agent, against)
        if not canonical.n or not other.n:
            return float("nan")
        return canonical.accuracy - other.accuracy

    def wording_gap(self, agent: str) -> float:
        """Same numbers, different nouns. Positive means the phrasing was load-bearing."""
        return self._gap(agent, "surface")

    def number_gap(self, agent: str) -> float:
        """Same phrasing structure, different numbers. Positive means the answer was recalled."""
        return self._gap(agent, "perturbed")

    def agents(self) -> list[str]:
        seen: list[str] = []
        for _, answer in self.rows:
            if answer.agent not in seen:
                seen.append(answer.agent)
        return seen

    def failures(self, agent: str | None = None) -> list[tuple[Item, Answer]]:
        return [
            (i, a)
            for i, a in self.rows
            if (agent is None or a.agent == agent) and not i.is_correct(a.value)
        ]

    def parse_of(self, item_id: str, agent: str) -> list[str]:
        """The equations an agent committed to for one item. Useful when debugging."""
        for item, answer in self.rows:
            if item.item_id == item_id and answer.agent == agent:
                step = answer.trace.last(StepKind.PARSE)
                if step:
                    return step.payload.get("spec", {}).get("equations", [])
        return []


def run_sweep(
    agents: Sequence[Agent],
    items: Iterable[Item],
    *,
    progress: Callable[[str], None] | None = None,
) -> SweepResult:
    """Run every agent over every item. Errors are recorded, never raised."""
    items = list(items)
    result = SweepResult()
    for agent in agents:
        for index, item in enumerate(items, 1):
            answer = agent.answer(item)
            result.add(item, answer)
            if progress:
                mark = "ok " if item.is_correct(answer.value) else "MISS"
                progress(f"{agent.name:<9} {index:>3}/{len(items)}  {mark}  {item.item_id}")
    return result
