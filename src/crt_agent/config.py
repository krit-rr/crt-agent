"""Settings, loaded from environment or `.env`.

Everything here has a default that works with no credentials at all, so `pytest`
and CI run the full pipeline against the mock provider without touching a network.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- model ---------------------------------------------------------------
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 1200
    temperature: float = 0.0
    llm_provider: str = "auto"  # auto | anthropic | mock

    # --- tracing -------------------------------------------------------------
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # --- storage -------------------------------------------------------------
    database_url: str = "sqlite:///crt_runs.db"

    # --- benchmark -----------------------------------------------------------
    seed: int = 20260825

    @property
    def use_mock(self) -> bool:
        if self.llm_provider == "mock":
            return True
        if self.llm_provider == "anthropic":
            return False
        return not self.anthropic_api_key

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
