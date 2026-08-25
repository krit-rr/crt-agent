"""The `claude -p` provider.

CI has no subscription, so the wire path is verified against a fake `claude`
executable that emits the same envelope shape the real one does. The envelope
fixtures below were captured from claude 2.1.245 / Haiku 4.5.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from crt_agent.llm.claude_cli import (
    ClaudeCLIError,
    ClaudeCLIProvider,
    contract_prompt,
    extract_json,
)
from crt_agent.llm.client import FORMALISE_TOOL

# Trimmed from a real run. Cache fields matter: they are most of the spend.
ENVELOPE = {
    "is_error": False,
    "session_id": "39737977-952c-5993-86ed-b0c6cf285f87",
    "total_cost_usd": 0.0151,
    "usage": {
        "input_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 32022,
        "output_tokens": 2177,
    },
    "result": "",
}

SPEC_JSON = {
    "variables": [
        {"name": "big", "description": "cost of the kettle"},
        {"name": "small", "description": "cost of the mug"},
    ],
    "equations": ["big + small = 3.40", "big - small = 3.00"],
    "query": "small",
}


def make_fake_claude(tmp_path, *, result: str, exit_code: int = 0, envelope_override=None):
    """Write a stub `claude` that records its argv and prints a canned envelope."""
    envelope = {**ENVELOPE, "result": result}
    if envelope_override:
        envelope.update(envelope_override)

    argv_log = tmp_path / "argv.json"
    script = tmp_path / "claude"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(argv_log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        # The real CLI writes warnings to stderr; anything reading merged streams breaks.
        "sys.stderr.write('Warning: no stdin data received in 3s\\n')\n"
        f"sys.stdout.write({json.dumps(json.dumps(envelope))})\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script, argv_log


@pytest.fixture
def provider_factory(tmp_path, monkeypatch):
    def build(result=json.dumps(SPEC_JSON), exit_code=0, envelope_override=None, **kwargs):
        script, argv_log = make_fake_claude(
            tmp_path, result=result, exit_code=exit_code, envelope_override=envelope_override
        )
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        return ClaudeCLIProvider(binary=str(script), retries=0, **kwargs), argv_log

    return build


# -- JSON extraction ------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the formalisation:\n\n{"a": 1}\n\nHope that helps.',
        '  \n {"a": 1} \n ',
    ],
)
def test_extract_json_survives_the_ways_models_wrap_output(reply):
    assert extract_json(reply) == {"a": 1}


@pytest.mark.parametrize("reply", ["no json here", "", "{unclosed", "{'single': 'quotes'}"])
def test_extract_json_refuses_to_guess(reply):
    """A broken reply must become an abstention, not a repaired guess."""
    with pytest.raises(ClaudeCLIError):
        extract_json(reply)


def test_contract_prompt_embeds_the_real_tool_schema():
    prompt = contract_prompt("SYSTEM TEXT", FORMALISE_TOOL)
    assert "SYSTEM TEXT" in prompt
    # The API path and the CLI path must describe the same schema.
    assert '"equations"' in prompt and '"query"' in prompt
    assert "variables, equations, query" in prompt


# -- the subprocess call --------------------------------------------------------------


def test_successful_call_parses_spec_and_accounts_for_cost(provider_factory):
    provider, _ = provider_factory()
    response = provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)

    spec = response.as_spec()
    assert spec.query == "small" and len(spec.equations) == 2
    assert response.cost_usd == pytest.approx(0.0151)
    # Cache reads are real spend and must not be reported as free.
    assert response.tokens_in == 10 + 0 + 32022
    assert response.tokens_out == 2177


def test_fenced_output_is_handled(provider_factory):
    provider, _ = provider_factory(result=f"```json\n{json.dumps(SPEC_JSON)}\n```")
    assert provider.call(system="s", prompt="p", tool=FORMALISE_TOOL).as_spec().query == "small"


def test_stderr_warnings_do_not_corrupt_the_envelope(provider_factory):
    """The stub writes to stderr like the real CLI does; stdout must stay clean JSON."""
    provider, _ = provider_factory()
    assert provider.call(system="s", prompt="p", tool=FORMALISE_TOOL).arguments == SPEC_JSON


def test_nonzero_exit_is_an_error(provider_factory):
    provider, _ = provider_factory(exit_code=2)
    with pytest.raises(ClaudeCLIError, match="exit 2"):
        provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)


def test_error_envelope_is_an_error(provider_factory):
    provider, _ = provider_factory(envelope_override={"is_error": True})
    with pytest.raises(ClaudeCLIError, match="reported an error"):
        provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)


def test_unparseable_reply_is_an_error(provider_factory):
    provider, _ = provider_factory(result="I think the mug costs twenty cents.")
    with pytest.raises(ClaudeCLIError, match="no JSON object"):
        provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)


# -- command construction -------------------------------------------------------------


def test_command_isolates_the_harness(provider_factory):
    provider, argv_log = provider_factory()
    provider.call(system="s", prompt="the question", tool=FORMALISE_TOOL)
    argv = json.loads(argv_log.read_text())

    assert argv[0] == "-p" and argv[1] == "the question"
    assert "--strict-mcp-config" in argv
    assert "--disallowed-tools" in argv
    assert argv[argv.index("--max-turns") + 1] == "1"
    assert argv[argv.index("--output-format") + 1] == "json"


def test_sessions_are_never_reused(provider_factory):
    """Cheaper, and forbidden: item N+1 would see item N and the benchmark would rot."""
    provider, argv_log = provider_factory()
    provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)
    argv = json.loads(argv_log.read_text())
    assert "--resume" not in argv and "--continue" not in argv and "-c" not in argv


def test_model_and_effort_are_passed_through(provider_factory):
    provider, argv_log = provider_factory(model="sonnet", effort="low")
    provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)
    argv = json.loads(argv_log.read_text())
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--effort") + 1] == "low"
    assert provider.name == "claude-cli:sonnet"


def test_missing_binary_gives_an_actionable_error():
    with pytest.raises(ClaudeCLIError, match="claude login"):
        ClaudeCLIProvider(binary="definitely-not-a-real-binary-xyz")


# -- the capability that gates the System 1 columns ------------------------------------


def test_provider_declares_it_cannot_sample_unreflectively(provider_factory):
    provider, _ = provider_factory()
    assert provider.supports_unreflective_sampling is False
