"""Application configuration — pydantic-settings with SecretStr, fail-fast validation."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Secrets never appear in repr, logs, or JSON
        secrets_dir=None,
    )

    # ── Cohere ────────────────────────────────────────────────────────────────
    cohere_api_key: SecretStr
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_rerank_model: str = "rerank-english-v3.0"
    cohere_generate_model: str = "command-r-plus"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rag_documents"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Postgres ──────────────────────────────────────────────────────────────
    database_url: SecretStr = SecretStr("postgresql+asyncpg://rag:rag@localhost:5432/rag")

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_audience: str = "rag-pipeline"
    jwt_issuer: str = "rag-pipeline-auth"

    # ── Application ───────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_rate_limit: int = 60          # requests per minute per API key
    max_context_tokens: int = 4096
    retrieval_top_k: int = 50
    rerank_top_n: int = 10
    hybrid_alpha: float = 0.5         # 0 = pure sparse, 1 = pure dense

    # ── Observability ─────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "rag-pipeline"

    @field_validator("jwt_secret_key")
    @classmethod
    def jwt_secret_must_be_strong(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            msg = "JWT_SECRET_KEY must be at least 32 characters"
            raise ValueError(msg)
        return v

    @field_validator("hybrid_alpha")
    @classmethod
    def alpha_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = "HYBRID_ALPHA must be in [0.0, 1.0]"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.app_env == "production":
            if self.qdrant_api_key is None:
                msg = "QDRANT_API_KEY is required in production"
                raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance; fails fast on misconfiguration."""
    return Settings()
