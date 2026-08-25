"""The model boundary.

Two implementations satisfy one protocol:

* `AnthropicProvider` — the real thing, via the Anthropic SDK, using a forced tool
  call so the model's output arrives as JSON that matches our Pydantic schema
  instead of prose we have to regex.
* `MockProvider` (see `crt_agent.llm.mock`) — deterministic, offline, and used by
  the test suite and CI.

Structured output is not a convenience here. `submit_formalisation` is defined so
there is no field in which the model *can* write an answer: it names variables and
equations, and that's all. The tool schema is the enforcement mechanism for the rule
that the model formalises and the solver computes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from crt_agent.config import Settings
from crt_agent.config import settings as default_settings
from crt_agent.schemas import ProblemSpec

# --------------------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------------------

FORMALISE_TOOL: dict[str, Any] = {
    "name": "submit_formalisation",
    "description": (
        "Submit a formal algebraic model of the word problem. Do not compute or state "
        "the answer: an external solver computes it from these equations. Declare every "
        "quantity you use, including intermediate ones."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "lowercase snake_case"},
                        "description": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["name", "description"],
                },
            },
            "equations": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Algebraic sentences using only the declared variable names, numbers, "
                    "and + - * / ( ) ^ . Exactly one '=' per equation. "
                    "Example: 'big + small = 1.10'"
                ),
            },
            "query": {"type": "string", "description": "name of the variable being asked for"},
            "unit": {"type": "string"},
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "modelling choices you committed to when reading the prose",
            },
        },
        "required": ["variables", "equations", "query"],
    },
}

INTUIT_TOOL: dict[str, Any] = {
    "name": "submit_snap_judgement",
    "description": (
        "Submit the first answer that comes to mind, without working anything out. "
        "This is a System 1 probe: speed over accuracy."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "gut_feel": {"type": "string", "description": "at most one short sentence"},
        },
        "required": ["value"],
    },
}

DIRECT_TOOL: dict[str, Any] = {
    "name": "submit_answer",
    "description": "Submit the final numeric answer along with your reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["value", "reasoning"],
    },
}


@dataclass
class LLMResponse:
    """One structured model call."""

    tool_name: str
    arguments: dict[str, Any]
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    #: Real spend for this call when the backend reports it (the CLI path does).
    #: Zero means "not reported", not "free".
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def as_spec(self) -> ProblemSpec:
        """Validate the model's formalisation. Raises `pydantic.ValidationError`."""
        return ProblemSpec.model_validate(self.arguments)


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    #: Can this backend return a genuinely unreflective answer?
    #:
    #: The dual-process arm needs one to measure System 1 at all. Backends that
    #: always deliberate before replying (notably the `claude -p` harness) set this
    #: False, and the report prints n/a for those columns rather than a number that
    #: looks like a System 1 measurement but isn't.
    supports_unreflective_sampling: bool

    def call(
        self,
        *,
        system: str,
        prompt: str,
        tool: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...


class AnthropicProvider:
    """Real model calls with a forced tool choice."""

    supports_unreflective_sampling = True

    def __init__(self, settings: Settings | None = None) -> None:
        from anthropic import Anthropic  # imported lazily so the mock path needs no SDK

        self.settings = settings or default_settings
        self.name = f"anthropic:{self.settings.model}"
        self._client = Anthropic(api_key=self.settings.anthropic_api_key)

    def call(
        self,
        *,
        system: str,
        prompt: str,
        tool: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        message = self._client.messages.create(
            model=self.settings.model,
            max_tokens=max_tokens or self.settings.max_tokens,
            temperature=self.settings.temperature if temperature is None else temperature,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )

        text_parts, tool_use = [], None
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use = block

        if tool_use is None:
            raise RuntimeError(f"model did not call {tool['name']}: {' '.join(text_parts)[:200]}")

        return LLMResponse(
            tool_name=tool_use.name,
            arguments=dict(tool_use.input),  # type: ignore[arg-type]
            text=" ".join(text_parts).strip(),
            tokens_in=message.usage.input_tokens,
            tokens_out=message.usage.output_tokens,
            raw=json.loads(message.model_dump_json()),
        )


def build_provider(settings: Settings | None = None, model: str | None = None) -> LLMProvider:
    """Pick a provider.

    `model` overrides the configured model, which is what the cross-model sweep uses
    to bind one provider instance per model.
    """
    cfg = settings or default_settings

    if cfg.llm_provider == "claude-cli":
        from crt_agent.llm.claude_cli import ClaudeCLIProvider

        return ClaudeCLIProvider(
            model=model or cfg.model,
            binary=cfg.claude_binary,
            timeout_s=cfg.claude_timeout_s,
            effort=cfg.claude_effort or None,
        )

    if cfg.use_mock:
        from crt_agent.llm.mock import MockProvider

        return MockProvider()

    if model:
        cfg = cfg.model_copy(update={"model": model})
    return AnthropicProvider(cfg)
