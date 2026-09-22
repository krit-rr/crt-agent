"""Any OpenAI-compatible chat endpoint: Ollama, LM Studio, vLLM, llama.cpp server.

This is the backend the benchmark has been waiting for. Every arm scores 100% on the
Claude family and the grounding delta is zero — the control solves CRT items unaided.
A 7-8B local model that still falls for the lures is the tier where the architecture
comparison becomes measurable, and that model speaks this protocol.

Two things worth knowing about it:

**Tool calling is optional and probed, not assumed.** Servers vary. If the server
honours `tool_choice`, we use it and get the forced-tool guarantee the Anthropic path
has. If it rejects tools (HTTP 4xx mentioning them, or returns prose), we fall back to
the same JSON contract `claude -p` uses. Either way only `variables`, `equations` and
`query` are read — the model still has no channel through which to state an answer.

**It can sample unreflectively.** Base chat completions do not deliberate before
replying, so `supports_unreflective_sampling = True` and the `dual` arm's System 1
columns are real measurements here — unlike the subscription route.

Cost is reported as zero because local inference has no metered price. The budget
guard treats zero as "not reported", so `--max-cost` is inert on this backend.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from crt_agent.config import Settings
from crt_agent.llm.client import LLMResponse
from crt_agent.llm.json_contract import contract_prompt, extract_json
from crt_agent.llm.registry import register_provider

#: (url, headers, body) -> decoded JSON response. Injectable so tests need no server.
Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class OpenAICompatError(RuntimeError):
    """The endpoint failed or returned something unparseable."""


def _urllib_transport(timeout_s: float) -> Transport:
    def send(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise OpenAICompatError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OpenAICompatError(
                f"could not reach {url}: {exc.reason}. Is the server running?"
            ) from exc

    return send


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic tool shape -> OpenAI function-calling shape. Same schema, new envelope."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


class OpenAICompatProvider:
    """One chat completion per call. Sessions are never reused — item independence."""

    supports_unreflective_sampling = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        timeout_s: float = 180.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = f"openai-compat:{model}"
        self._send = transport or _urllib_transport(timeout_s)
        #: None = not yet probed. Set on first call and remembered for the sweep.
        self._tools_supported: bool | None = None

    # -- request construction --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body(
        self,
        system: str,
        prompt: str,
        tool: dict[str, Any],
        *,
        use_tools: bool,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        system_text = system if use_tools else contract_prompt(system, tool)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0 if temperature is None else temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if use_tools:
            body["tools"] = [_to_openai_tool(tool)]
            body["tool_choice"] = {"type": "function", "function": {"name": tool["name"]}}
        return body

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
        url = f"{self.base_url}/chat/completions"
        kwargs = {"max_tokens": max_tokens, "temperature": temperature}

        if self._tools_supported is not False:
            try:
                body = self._body(system, prompt, tool, use_tools=True, **kwargs)
                data = self._send(url, self._headers(), body)
                arguments = self._arguments_from_tool_call(data, tool)
                if arguments is not None:
                    self._tools_supported = True
                    return self._response(data, tool, arguments)
            except OpenAICompatError as exc:
                if self._tools_supported is True or "tool" not in str(exc).lower():
                    raise
            # Server ignored or rejected tools: remember, and fall through to the contract.
            self._tools_supported = False

        body = self._body(system, prompt, tool, use_tools=False, **kwargs)
        data = self._send(url, self._headers(), body)
        text = self._message_text(data)
        return self._response(data, tool, extract_json(text, error=OpenAICompatError))

    # -- response parsing ----------------------------------------------------------

    @staticmethod
    def _message(data: dict[str, Any]) -> dict[str, Any]:
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatError(f"no choices in response: {str(data)[:200]!r}") from exc

    @classmethod
    def _message_text(cls, data: dict[str, Any]) -> str:
        return cls._message(data).get("content") or ""

    @classmethod
    def _arguments_from_tool_call(
        cls, data: dict[str, Any], tool: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The forced tool's arguments, or None if the server answered in prose."""
        calls = cls._message(data).get("tool_calls") or []
        for call in calls:
            fn = call.get("function", {})
            if fn.get("name") == tool["name"]:
                raw = fn.get("arguments", "{}")
                try:
                    return json.loads(raw) if isinstance(raw, str) else dict(raw)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(f"tool arguments were not JSON: {exc}") from exc
        return None

    def _response(
        self, data: dict[str, Any], tool: dict[str, Any], arguments: dict[str, Any]
    ) -> LLMResponse:
        usage = data.get("usage") or {}
        return LLMResponse(
            tool_name=tool["name"],
            arguments=arguments,
            text="",
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=0.0,  # local inference: unmetered, and the budget guard knows 0 = unreported
            raw={"model": data.get("model"), "usage": usage, "tools": self._tools_supported},
        )


@register_provider("openai-compat")
def _openai_compat_factory(cfg: Settings, model: str | None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        model=model or cfg.model,
        base_url=cfg.openai_base_url,
        api_key=cfg.openai_api_key,
        timeout_s=cfg.openai_timeout_s,
    )
