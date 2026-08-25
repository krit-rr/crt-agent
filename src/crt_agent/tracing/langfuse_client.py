"""LangFuse wiring, with a no-op fallback.

Observability should never be load-bearing for correctness. If LangFuse is not
configured (or the container isn't up), `get_tracer()` returns a null tracer and
everything else behaves identically — the local `Trace` object is the source of
truth for the benchmark, and LangFuse is where you go to *look* at a run.

Spans are nested one level: a root span per item attempt, a child span per graph
node, with the node's payload as span input/output.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Protocol

from crt_agent.config import Settings
from crt_agent.config import settings as default_settings


class Tracer(Protocol):
    @contextlib.contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[Any]: ...

    def flush(self) -> None: ...


class NullTracer:
    """Used when LangFuse is not configured. Zero dependencies, zero side effects."""

    enabled = False

    @contextlib.contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[None]:
        yield None

    def flush(self) -> None:
        return None


class LangfuseTracer:
    """Thin adapter over the LangFuse client."""

    enabled = True

    def __init__(self, settings: Settings) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

    @contextlib.contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[Any]:
        with self._client.start_as_current_observation(name=name, as_type="span", **kwargs) as span:
            yield span

    def flush(self) -> None:
        self._client.flush()


def get_tracer(settings: Settings | None = None) -> Tracer:
    cfg = settings or default_settings
    if not cfg.tracing_enabled:
        return NullTracer()
    try:
        return LangfuseTracer(cfg)
    except Exception:  # noqa: BLE001 - observability must never break a run
        return NullTracer()
