"""The provider registry and the OpenAI-compatible backend.

No server in CI, so the wire path runs against a fake transport that returns the
same response shapes Ollama / vLLM do. Two shapes matter: a server that honours
`tool_choice`, and one that ignores tools and answers in prose.
"""

from __future__ import annotations

import json

import pytest

from crt_agent.config import Settings
from crt_agent.llm.client import FORMALISE_TOOL, LLMProvider, build_provider
from crt_agent.llm.openai_compat import OpenAICompatError, OpenAICompatProvider
from crt_agent.llm.registry import _REGISTRY, registered_providers, resolve_provider_name

SPEC_JSON = {
    "variables": [
        {"name": "big", "description": "cost of the kettle"},
        {"name": "small", "description": "cost of the mug"},
    ],
    "equations": ["big + small = 3.40", "big - small = 3.00"],
    "query": "small",
}

# -- registry --------------------------------------------------------------------------


def test_every_backend_is_registered():
    assert registered_providers() == ["anthropic", "claude-cli", "mock", "openai-compat"]


def test_unknown_provider_fails_loudly_and_lists_what_exists():
    with pytest.raises(ValueError, match="registered:.*openai-compat"):
        build_provider(Settings(llm_provider="nope"))


@pytest.mark.parametrize(
    ("key", "name", "expected"),
    [("", "auto", "mock"), ("sk-x", "auto", "anthropic"), ("sk-x", "mock", "mock")],
)
def test_auto_rule_lives_in_one_place(key, name, expected):
    assert resolve_provider_name(Settings(anthropic_api_key=key, llm_provider=name)) == expected


def test_openai_compat_is_selectable_by_name():
    provider = build_provider(Settings(llm_provider="openai-compat", model="qwen2.5:7b"))
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.name == "openai-compat:qwen2.5:7b"


def test_registered_factories_satisfy_the_protocol():
    """Guard the one boundary instead of auditing every call site."""
    registered_providers()
    for name in ("mock", "openai-compat"):  # the ones constructible with no credentials
        provider = _REGISTRY[name](Settings(llm_provider=name), None)
        assert isinstance(provider, LLMProvider)
        assert isinstance(provider.supports_unreflective_sampling, bool)


# -- openai-compat: fake transport ------------------------------------------------------


class FakeServer:
    """Records requests; answers with tool calls or prose depending on `tools_ok`."""

    def __init__(self, *, tools_ok: bool, reply: str | None = None) -> None:
        self.tools_ok = tools_ok
        self.reply = reply
        self.requests: list[dict] = []

    def __call__(self, url, headers, body):
        self.requests.append({"url": url, "headers": headers, "body": body})
        usage = {"prompt_tokens": 120, "completion_tokens": 40}
        if "tools" in body and self.tools_ok:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": body["tools"][0]["function"]["name"],
                            "arguments": json.dumps(SPEC_JSON),
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": self.reply or json.dumps(SPEC_JSON)}
        return {"model": body["model"], "choices": [{"message": message}], "usage": usage}


def _provider(server: FakeServer) -> OpenAICompatProvider:
    return OpenAICompatProvider("qwen2.5:7b", transport=server, base_url="http://x/v1/")


def test_forced_tool_path_when_the_server_supports_it():
    server = FakeServer(tools_ok=True)
    resp = _provider(server).call(system="SYS", prompt="Q", tool=FORMALISE_TOOL)
    assert resp.as_spec().query == "small"
    assert resp.tokens_in == 120 and resp.cost_usd == 0.0
    body = server.requests[0]["body"]
    assert body["tool_choice"]["function"]["name"] == FORMALISE_TOOL["name"]
    assert body["messages"][0]["content"] == "SYS"  # no contract text on the tool path
    assert server.requests[0]["url"] == "http://x/v1/chat/completions"


def test_falls_back_to_json_contract_when_tools_are_ignored():
    """Prose back from a tools request => retry under the JSON contract, then remember."""
    server = FakeServer(tools_ok=False)
    provider = _provider(server)
    resp = provider.call(system="SYS", prompt="Q", tool=FORMALISE_TOOL)
    assert resp.as_spec().equations == SPEC_JSON["equations"]
    assert len(server.requests) == 2
    assert "OUTPUT CONTRACT" in server.requests[1]["body"]["messages"][0]["content"]
    assert "tools" not in server.requests[1]["body"]

    provider.call(system="SYS", prompt="Q2", tool=FORMALISE_TOOL)
    assert len(server.requests) == 3  # no second probe: the answer is remembered


def test_fenced_json_on_the_contract_path_is_fine():
    server = FakeServer(tools_ok=False, reply=f"```json\n{json.dumps(SPEC_JSON)}\n```")
    assert _provider(server).call(system="S", prompt="Q", tool=FORMALISE_TOOL).as_spec()


def test_prose_with_no_json_is_a_hard_error_not_a_guess():
    server = FakeServer(tools_ok=False, reply="the mug costs twenty cents")
    with pytest.raises(OpenAICompatError, match="no JSON object"):
        _provider(server).call(system="S", prompt="Q", tool=FORMALISE_TOOL)


def test_the_answer_has_no_field_on_this_backend_either():
    """The architectural guarantee is backend-independent: only the schema's fields
    are ever read. An extra `answer` key from a chatty model is rejected upstream by
    the Pydantic model, exactly as on the API path."""
    chatty = {**SPEC_JSON, "answer": 0.2}
    server = FakeServer(tools_ok=False, reply=json.dumps(chatty))
    resp = _provider(server).call(system="S", prompt="Q", tool=FORMALISE_TOOL)
    spec = resp.as_spec()
    assert not hasattr(spec, "answer")
    assert spec.query == "small"


def test_bearer_header_only_when_a_key_is_set():
    server = FakeServer(tools_ok=True)
    OpenAICompatProvider("m", transport=server).call(system="S", prompt="Q", tool=FORMALISE_TOOL)
    assert "Authorization" not in server.requests[0]["headers"]
    OpenAICompatProvider("m", transport=server, api_key="k").call(
        system="S", prompt="Q", tool=FORMALISE_TOOL
    )
    assert server.requests[1]["headers"]["Authorization"] == "Bearer k"
