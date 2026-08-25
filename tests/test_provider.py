"""Provider selection and Anthropic response handling.

The Anthropic path is the one piece of this repo that CI cannot exercise for real, so
it is exercised against a stubbed SDK client instead. This is the code that breaks
first when someone finally sets an API key, and it should not break silently.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from crt_agent.config import Settings
from crt_agent.llm.client import FORMALISE_TOOL, AnthropicProvider, build_provider
from crt_agent.llm.mock import MockProvider


def _message(blocks, tokens_in=120, tokens_out=45):
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
        model_dump_json=lambda: '{"stub": true}',
    )


class _StubClient:
    """Captures the request and returns a canned message."""

    def __init__(self, message):
        self._message = message
        self.last_kwargs: dict = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._message


@pytest.fixture
def anthropic_provider(monkeypatch):
    def build(blocks):
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.settings = Settings(anthropic_api_key="sk-test", model="claude-test")
        provider.name = "anthropic:claude-test"
        provider._client = _StubClient(_message(blocks))
        return provider

    return build


def test_tool_use_block_is_parsed_into_a_validated_spec(anthropic_provider):
    provider = anthropic_provider(
        [
            SimpleNamespace(type="text", text="thinking out loud"),
            SimpleNamespace(
                type="tool_use",
                name="submit_formalisation",
                input={
                    "variables": [
                        {"name": "big", "description": "bat"},
                        {"name": "small", "description": "ball"},
                    ],
                    "equations": ["big + small = 1.10", "big - small = 1.00"],
                    "query": "small",
                },
            ),
        ]
    )
    response = provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)

    assert response.tool_name == "submit_formalisation"
    assert response.tokens_in == 120 and response.tokens_out == 45
    assert response.text == "thinking out loud"

    spec = response.as_spec()
    assert spec.query == "small" and len(spec.equations) == 2


def test_the_tool_call_is_forced(anthropic_provider):
    provider = anthropic_provider(
        [
            SimpleNamespace(
                type="tool_use",
                name="submit_formalisation",
                input={
                    "variables": [{"name": "x", "description": "x"}],
                    "equations": ["x = 1"],
                    "query": "x",
                },
            )
        ]
    )
    provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)
    sent = provider._client.last_kwargs

    # If tool_choice is ever relaxed, the model regains a channel for stating an
    # answer in prose — which is the one thing this architecture forbids.
    assert sent["tool_choice"] == {"type": "tool", "name": "submit_formalisation"}
    assert sent["model"] == "claude-test"
    assert [t["name"] for t in sent["tools"]] == ["submit_formalisation"]


def test_temperature_override_reaches_the_sdk(anthropic_provider):
    provider = anthropic_provider(
        [SimpleNamespace(type="tool_use", name="submit_snap_judgement", input={"value": 0.1})]
    )
    provider.call(system="s", prompt="p", tool=FORMALISE_TOOL, temperature=1.0)
    assert provider._client.last_kwargs["temperature"] == 1.0


def test_a_text_only_reply_is_a_hard_error(anthropic_provider):
    """Better to fail loudly than to fall back to scraping a number out of prose."""
    provider = anthropic_provider([SimpleNamespace(type="text", text="the ball costs $0.05")])
    with pytest.raises(RuntimeError, match="did not call"):
        provider.call(system="s", prompt="p", tool=FORMALISE_TOOL)


def test_provider_selection_falls_back_to_mock_without_a_key():
    assert isinstance(build_provider(Settings(anthropic_api_key="")), MockProvider)
    assert isinstance(
        build_provider(Settings(anthropic_api_key="sk-x", llm_provider="mock")), MockProvider
    )


def test_explicit_anthropic_provider_is_honoured(monkeypatch):
    created = {}

    class _Anthropic:
        def __init__(self, api_key):
            created["api_key"] = api_key

    monkeypatch.setattr("anthropic.Anthropic", _Anthropic)
    provider = build_provider(Settings(anthropic_api_key="sk-x", llm_provider="anthropic"))
    assert isinstance(provider, AnthropicProvider)
    assert created["api_key"] == "sk-x"
