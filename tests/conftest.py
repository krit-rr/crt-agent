from __future__ import annotations

import pytest

from crt_agent.config import Settings
from crt_agent.llm.mock import MockProvider
from crt_agent.tracing.langfuse_client import NullTracer


@pytest.fixture(scope="session")
def provider() -> MockProvider:
    return MockProvider()


@pytest.fixture(scope="session")
def tracer() -> NullTracer:
    return NullTracer()


@pytest.fixture
def offline_settings() -> Settings:
    return Settings(
        anthropic_api_key="",
        llm_provider="mock",
        langfuse_public_key="",
        langfuse_secret_key="",
        database_url="sqlite:///:memory:",
    )
