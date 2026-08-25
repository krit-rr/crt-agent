"""Cross-model sweeps: the same four architectures, several models.

Running one strong model is the least informative thing you can do with this
benchmark. A frontier model solves CRT items unaided, so every arm lands near 100%,
the control arm stops being a control, and the result says nothing about the
scaffolding — the ceiling has hidden the effect you were trying to measure.

The measurement lives in the *interaction* between model strength and architecture.
Weak models fail CRT items in exactly the way humans do (straight into the lure), and
grounding rescues them; strong models don't need rescuing. So the quantity worth
reporting is not accuracy but:

    grounding delta = accuracy(tool) - accuracy(ablation)      per model

which should be large and shrinking as models get stronger. "Grounding is worth +54
points on Haiku and +2 on Opus" is a finding. "The tool agent scored 98%" is not.

A practical consequence: prefer the cheapest model you have access to when you want
the effect to be visible, and include a strong one only to show the delta closing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from crt_agent.agents.base import Agent
from crt_agent.bench.runner import Metrics, SweepResult, run_sweep
from crt_agent.schemas import Item, ItemSet


@dataclass
class MatrixResult:
    """One `SweepResult` per model, plus the cross-model views."""

    sweeps: dict[str, SweepResult] = field(default_factory=dict)
    #: models in the order they were run
    models: list[str] = field(default_factory=list)

    def add(self, model: str, sweep: SweepResult) -> None:
        if model not in self.sweeps:
            self.models.append(model)
        self.sweeps[model] = sweep

    def agents(self) -> list[str]:
        seen: list[str] = []
        for model in self.models:
            for agent in self.sweeps[model].agents():
                if agent not in seen:
                    seen.append(agent)
        return seen

    def metrics(self, model: str, agent: str, item_set: ItemSet | None = None) -> Metrics:
        return self.sweeps[model].metrics(agent, item_set)

    def grounding_delta(
        self, model: str, *, grounded: str = "tool", control: str = "ablation"
    ) -> float:
        """accuracy(grounded arm) - accuracy(control arm) for one model.

        This is the headline number of a cross-model sweep: what the scaffolding is
        worth, for this model, on this item set.
        """
        sweep = self.sweeps[model]
        if grounded not in sweep.agents() or control not in sweep.agents():
            return float("nan")
        return sweep.metrics(grounded).accuracy - sweep.metrics(control).accuracy

    def total_cost(self) -> float:
        return sum(
            self.sweeps[m].metrics(a).cost_usd for m in self.models for a in self.sweeps[m].agents()
        )


def run_matrix(
    models: Sequence[str],
    build_agents: Callable[[str], list[Agent]],
    items: Sequence[Item],
    *,
    progress: Callable[[str], None] | None = None,
    concurrency: int = 1,
    max_cost_usd: float | None = None,
) -> MatrixResult:
    """Run the full arm sweep once per model.

    `build_agents` is a factory rather than a list because each model needs its own
    provider instance bound to it. The budget ceiling is shared across the whole
    matrix and consumed in model order, so a run that overruns degrades by dropping
    later models entirely rather than by half-measuring all of them.
    """
    matrix = MatrixResult()
    remaining = max_cost_usd

    for model in models:
        if remaining is not None and remaining <= 0:
            if progress:
                progress(f"BUDGET  skipping {model}: ceiling reached")
            continue
        if progress:
            progress(f"=== model: {model} ===")

        sweep = run_sweep(
            build_agents(model),
            items,
            progress=progress,
            concurrency=concurrency,
            max_cost_usd=remaining,
        )
        matrix.add(model, sweep)

        if remaining is not None:
            spent = sum(sweep.metrics(a).cost_usd for a in sweep.agents())
            remaining -= spent

    return matrix
