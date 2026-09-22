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
    # auto | anthropic | claude-cli | openai-compat | mock  (see llm/registry.py)
    #   auto          -> anthropic if an API key is set, else mock
    #   claude-cli    -> `claude -p`, billed to your Pro/Max subscription (no API key)
    #   openai-compat -> any OpenAI-compatible endpoint; the local-model route
    llm_provider: str = "auto"
    claude_binary: str = "claude"
    claude_timeout_s: float = 180.0
    claude_effort: str = ""
    # OpenAI-compatible endpoint (Ollama default). The key is optional for local servers.
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""
    openai_timeout_s: float = 180.0

    # --- tracing -------------------------------------------------------------
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # --- storage -------------------------------------------------------------
    database_url: str = "sqlite:///crt_runs.db"

    # --- benchmark -----------------------------------------------------------
    seed: int = 20260825

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
