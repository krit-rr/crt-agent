"""A deterministic stand-in for the model.

Its job is to make the *whole* pipeline — graphs, tracing, auditing, storage,
metrics — runnable in CI with no API key and no network, while still producing a
benchmark that isn't trivially perfect.

It behaves like a plausible model rather than an oracle:

* **Formalising** it does well, via the rule-based matcher, but with a small,
  deterministic error rate: on some items it emits a *well-formed spec of the wrong
  problem* (the lure formalisation). That is exactly the failure mode a real parser
  has, and it is the one the trace auditor cannot catch — which is worth being able
  to demonstrate.
* **Snap judgements** are the lure, always. That is what System 1 is for.
* **Unaided direct answers** fall for the lure most of the time, more often on
  perturbed items than canonical ones — mirroring the recall-vs-reason gap the
  benchmark exists to measure.

Everything is keyed off a hash of the prompt, so runs are reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Any

from crt_agent.agents.matcher import match_spec
from crt_agent.config import Settings
from crt_agent.items.templates import TEMPLATES
from crt_agent.llm.client import LLMResponse
from crt_agent.llm.registry import register_provider


def _bucket(text: str, salt: str) -> float:
    """Stable pseudo-random number in [0, 1) for this prompt."""
    digest = hashlib.sha256(f"{salt}::{text}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _canonical_wording(text: str) -> bool:
    """Is this one of the wordings that saturates pretraining corpora?"""
    return any(
        marker in text
        for marker in (
            "A bat and a ball cost $1.10",
            "5 machines take 5 minutes to make 5 widgets",
            "patch of lily pads",
        )
    )


class MockProvider:
    """Offline provider. Same interface as `AnthropicProvider`."""

    name = "mock"
    supports_unreflective_sampling = True

    #: fraction of items where the parser emits a well-formed spec of the WRONG problem
    misparse_rate: float = 0.08

    def call(
        self,
        *,
        system: str,
        prompt: str,
        tool: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        handler = {
            "submit_formalisation": self._formalise,
            "submit_snap_judgement": self._intuit,
            "submit_answer": self._direct,
        }[tool["name"]]
        args = handler(prompt)
        return LLMResponse(
            tool_name=tool["name"],
            arguments=args,
            text="",
            tokens_in=len(prompt) // 4,
            tokens_out=60,
        )

    # -- handlers ----------------------------------------------------------------

    def _formalise(self, prompt: str) -> dict[str, Any]:
        matched = match_spec(prompt)
        if matched is None:
            # An unrecognised phrasing: emit a spec that will fail validation the way
            # a confused parser would, rather than silently inventing one.
            return {
                "variables": [{"name": "answer", "description": "the requested quantity"}],
                "equations": ["answer = answer"],
                "query": "answer",
                "assumptions": ["mock provider could not formalise this phrasing"],
            }

        family, spec = matched
        if _bucket(prompt, "misparse") < self.misparse_rate:
            spec = self._lure_spec(family, spec)
        return spec.model_dump()

    def _lure_spec(self, family: str, spec: Any) -> Any:
        """A well-formed formalisation of the trap instead of the problem."""
        from crt_agent.schemas import ProblemSpec, Variable

        if family == "bat_ball":
            # "just subtract the difference from the total, and stop"
            total = spec.equations[0].split("=")[1].strip()
            diff = spec.equations[1].split("=")[1].strip()
            return ProblemSpec(
                variables=[Variable(name="small", description="cost of the smaller item")],
                equations=[f"small = {total} - {diff}"],
                query="small",
                unit=spec.unit,
                assumptions=["(mock) treated the difference as the whole computation"],
            )
        # Generic degradation: drop the last equation, which usually under-determines
        # the system and surfaces as a solver error rather than a silent wrong answer.
        return spec.model_copy(update={"equations": spec.equations[:1]})

    def _intuit(self, prompt: str) -> dict[str, Any]:
        value = self._lure_value(prompt)
        return {"value": value, "gut_feel": "the numbers in the sentence line up that way"}

    def _direct(self, prompt: str) -> dict[str, Any]:
        lure = self._lure_value(prompt)
        answer = self._true_value(prompt)
        if answer is None:
            return {"value": lure, "reasoning": "(mock) pattern-matched from the surface numbers"}
        # Unaided reflection succeeds more often on wordings it has memorised.
        threshold = 0.80 if _canonical_wording(prompt) else 0.35
        reflected = _bucket(prompt, "reflect") < threshold
        note = "(mock) worked it through" if reflected else "(mock) went with the obvious read"
        return {"value": answer if reflected else lure, "reasoning": note}

    # -- helpers -----------------------------------------------------------------

    def _params(self, prompt: str):
        matched = match_spec(prompt)
        if matched is None:
            return None
        from crt_agent.agents.matcher import PATTERNS
        from crt_agent.agents.matcher import _params as extract

        family, _ = matched
        m = PATTERNS[family].search(prompt)
        params = dict(TEMPLATES[family].canonical)
        params.update(extract(family, m))  # type: ignore[arg-type]
        return family, params

    def _lure_value(self, prompt: str) -> float:
        got = self._params(prompt)
        if got is None:
            return 0.0
        family, params = got
        return float(TEMPLATES[family].lure(params))

    def _true_value(self, prompt: str) -> float | None:
        got = self._params(prompt)
        if got is None:
            return None
        family, params = got
        return float(TEMPLATES[family].answer(params))


@register_provider("mock")
def _mock_factory(cfg: Settings, model: str | None) -> MockProvider:
    return MockProvider()
