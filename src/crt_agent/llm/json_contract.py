"""The JSON-contract fallback for backends without API-level forced tool use.

Two backends need it: `claude -p` (an agent harness, no `tool_choice`) and any
OpenAI-compatible local server that doesn't implement tool calling. Rather than each
carrying its own copy, the contract lives here so it cannot drift.

The tool definitions in `llm/client.py` remain the single source of truth: the API
path passes them to `tools=[...]`; this path compiles the same `input_schema` into
prose. The reply is then validated by the same Pydantic model — so the architectural
guarantee is untouched. Only `variables`, `equations` and `query` are ever read; the
model still has no channel through which to state an answer.
"""

from __future__ import annotations

import json
from typing import Any


class ContractError(RuntimeError):
    """The reply did not honour the JSON contract."""


def contract_prompt(system: str, tool: dict[str, Any]) -> str:
    """Render a tool's `input_schema` into a JSON-output contract."""
    schema = json.dumps(tool["input_schema"], indent=2)
    required = ", ".join(tool["input_schema"].get("required", []))
    return (
        f"{system}\n\n"
        "OUTPUT CONTRACT\n"
        "Reply with exactly ONE JSON object and nothing else. No preamble, no "
        "explanation, no markdown fences.\n\n"
        f"It must validate against this JSON Schema:\n{schema}\n\n"
        f"Required fields: {required}\n"
    )


def extract_json(text: str, *, error: type[Exception] = ContractError) -> dict[str, Any]:
    """Pull one JSON object out of a model reply.

    Models fence their JSON roughly half the time regardless of instructions, so this
    strips fences and falls back to the outermost brace pair. It deliberately does not
    try to repair malformed JSON — a broken reply should surface as an abstention, not
    as a guess about what the model meant.

    `error` lets a backend raise its own exception class so callers can keep catching
    the type they already know about.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else body
        if body.endswith("```"):
            body = body[: body.rindex("```")]
        body = body.strip()
        if body.startswith("json"):
            body = body[4:].strip()

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        raise error(f"no JSON object in reply: {text[:200]!r}")
    try:
        return json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise error(f"malformed JSON in reply: {exc}") from exc
