"""Claude via the `claude -p` CLI, authenticated by a Pro/Max subscription.

Anthropic supports driving Claude programmatically through the Agent SDK / `claude -p`
with subscription auth rather than an API key. Usage is metered against a separate
monthly Agent SDK credit pool. That makes this a legitimate way to run the benchmark
with no API key and no billing setup.

It is not equivalent to the API path, and the differences are load-bearing enough that
they are encoded in code rather than left in a footnote:

**1. No API-level forced tool use.** `claude -p` is an agent harness, not the Messages
API, so `tool_choice` isn't available. The tool's `input_schema` is instead rendered
into the system prompt as a JSON contract, and the reply is parsed and validated by the
same Pydantic model. The *architectural* guarantee survives intact — we read only
`variables`, `equations` and `query`, so the model still has no channel through which
to state an answer. What degrades is reliability: malformed replies become abstentions
rather than schema retries.

**2. It cannot sample an unreflective answer.** The harness reasons before replying no
matter what; at `--effort low` Haiku still spent 753 thinking tokens on the bat-and-ball
item and returned $0.05 rather than the $0.10 lure. A "snap judgement" that has been
deliberated over is not a System 1 measurement, so this provider declares
`supports_unreflective_sampling = False` and the report prints `n/a` for those columns
instead of a misleading number.

**3. Cost is dominated by harness overhead.** Every call re-sends Claude Code's ~30k
token system prompt. The first call in a session pays full price (~$0.06 on Haiku);
later ones hit the prompt cache and drop to ~$0.006-0.02. Session *reuse* would cut
this further and is deliberately not done — see `_build_command`.

Measured on claude 2.1.245 / Haiku 4.5, August 2026.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from crt_agent.config import Settings
from crt_agent.llm.client import LLMResponse
from crt_agent.llm.json_contract import contract_prompt as _contract_prompt
from crt_agent.llm.json_contract import extract_json as _extract_json
from crt_agent.llm.registry import register_provider

#: Tools Claude Code must not touch. The model is here to emit JSON, not to act.
_DENIED_TOOLS = "Bash Edit Write Read Glob Grep WebSearch WebFetch Task NotebookEdit TodoWrite"


class ClaudeCLIError(RuntimeError):
    """The CLI failed, timed out, or returned something unparseable."""


def contract_prompt(system: str, tool: dict[str, Any]) -> str:
    """See `crt_agent.llm.json_contract` — shared with the OpenAI-compatible path."""
    return _contract_prompt(system, tool)


def extract_json(text: str) -> dict[str, Any]:
    """See `crt_agent.llm.json_contract`. Raises `ClaudeCLIError` on a bad reply."""
    return _extract_json(text, error=ClaudeCLIError)


class ClaudeCLIProvider:
    """Runs each call as a fresh `claude -p` subprocess."""

    #: See module docstring, point 2. The dual-process arm reads this.
    supports_unreflective_sampling = False

    def __init__(
        self,
        model: str = "haiku",
        *,
        binary: str = "claude",
        timeout_s: float = 180.0,
        retries: int = 1,
        effort: str | None = None,
    ) -> None:
        resolved = shutil.which(binary)
        if resolved is None:
            raise ClaudeCLIError(
                f"{binary!r} is not on PATH. Install Claude Code and run `claude login` "
                "to authenticate with your Pro/Max subscription."
            )
        self.binary = resolved
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self.effort = effort
        self.name = f"claude-cli:{model}"

    # -- command construction ------------------------------------------------------

    def _build_command(self, system: str, prompt: str, tool: dict[str, Any]) -> list[str]:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--max-turns",
            "1",
            # Keep the harness from reaching for anything: no MCP servers from the
            # user's config, no tools, one turn.
            "--strict-mcp-config",
            "--disallowed-tools",
            _DENIED_TOOLS,
            "--system-prompt",
            contract_prompt(system, tool),
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        # NOTE: deliberately no --resume / --continue. Reusing one session would let
        # the prompt cache amortise the ~30k-token harness prefix and cut cost several
        # fold, but every item after the first would then see the previous items in
        # context. Item independence is the premise of the whole benchmark, so the
        # cost stays.
        return cmd

    # -- the call ------------------------------------------------------------------

    def call(
        self,
        *,
        system: str,
        prompt: str,
        tool: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        cmd = self._build_command(system, prompt, tool)
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                return self._run_once(cmd, tool)
            except ClaudeCLIError as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        raise ClaudeCLIError(f"{self.name} failed after {self.retries + 1} attempts: {last_error}")

    def _run_once(self, cmd: list[str], tool: dict[str, Any]) -> LLMResponse:
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                # Without this the CLI waits ~3s per call for stdin that never comes.
                # Across a full sweep that alone is ~13 minutes of pure stall.
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(f"timed out after {self.timeout_s}s") from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()[:300]
            raise ClaudeCLIError(f"exit {completed.returncode}: {stderr}")

        # stdout carries the envelope; warnings go to stderr and must not be merged.
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCLIError(f"envelope was not JSON: {completed.stdout[:200]!r}") from exc

        if envelope.get("is_error"):
            raise ClaudeCLIError(f"claude reported an error: {envelope.get('result')!r}")

        arguments = extract_json(envelope.get("result", ""))
        usage = envelope.get("usage", {}) or {}

        return LLMResponse(
            tool_name=tool["name"],
            arguments=arguments,
            text="",
            # Cache reads and creations are real spend and belong in the token counts,
            # otherwise the harness overhead looks free.
            tokens_in=(
                int(usage.get("input_tokens", 0))
                + int(usage.get("cache_creation_input_tokens", 0))
                + int(usage.get("cache_read_input_tokens", 0))
            ),
            tokens_out=int(usage.get("output_tokens", 0)),
            cost_usd=float(envelope.get("total_cost_usd", 0.0) or 0.0),
            raw={"session_id": envelope.get("session_id"), "usage": usage},
        )


@register_provider("claude-cli")
def _claude_cli_factory(cfg: Settings, model: str | None) -> ClaudeCLIProvider:
    return ClaudeCLIProvider(
        model=model or cfg.model,
        binary=cfg.claude_binary,
        timeout_s=cfg.claude_timeout_s,
        effort=cfg.claude_effort or None,
    )
