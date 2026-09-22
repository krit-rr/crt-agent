"""Provider registry.

`LLM_PROVIDER` is resolved here and nowhere else. Each backend registers its own
factory at the bottom of its own module, so adding a backend touches exactly one
file — the guarantee that made the OpenAI-compatible adapter a single new module.

Registered names:

    anthropic      Anthropic Messages API, forced tool use, needs ANTHROPIC_API_KEY
    claude-cli     `claude -p` on a Pro/Max subscription, JSON contract, no key
    openai-compat  any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, ...)
    mock           deterministic offline provider used by tests and CI
    auto           anthropic if a key is set, else mock  (the default)
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

from crt_agent.config import Settings
from crt_agent.config import settings as default_settings

if TYPE_CHECKING:  # avoids a circular import; providers import LLMResponse from client
    from crt_agent.llm.client import LLMProvider

ProviderFactory = Callable[[Settings, "str | None"], "LLMProvider"]

_REGISTRY: dict[str, ProviderFactory] = {}

#: Modules that register a provider on import. Listed explicitly (not discovered) so
#: a typo in a module name fails at import time rather than as "unknown provider".
_PROVIDER_MODULES = (
    "crt_agent.llm.client",
    "crt_agent.llm.mock",
    "crt_agent.llm.claude_cli",
    "crt_agent.llm.openai_compat",
)


def register_provider(name: str) -> Callable[[ProviderFactory], ProviderFactory]:
    """Decorator: `@register_provider("name")` on a `(settings, model) -> provider` factory."""

    def deco(factory: ProviderFactory) -> ProviderFactory:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"provider {name!r} registered twice")
        _REGISTRY[name] = factory
        return factory

    return deco


def resolve_provider_name(cfg: Settings) -> str:
    """The one place the `auto` rule lives."""
    if cfg.llm_provider != "auto":
        return cfg.llm_provider
    return "anthropic" if cfg.anthropic_api_key else "mock"


def registered_providers() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def build_provider(settings: Settings | None = None, model: str | None = None) -> LLMProvider:
    """Pick a provider.

    `model` overrides the configured model, which is what the cross-model sweep uses
    to bind one provider instance per model.
    """
    cfg = settings or default_settings
    name = resolve_provider_name(cfg)
    _load_all()
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown LLM_PROVIDER={name!r}; registered: {sorted(_REGISTRY)}"
        ) from None
    return factory(cfg, model)


def _load_all() -> None:
    for module in _PROVIDER_MODULES:
        import_module(module)
