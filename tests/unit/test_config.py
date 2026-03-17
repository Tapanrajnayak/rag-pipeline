"""Unit tests for configuration — SecretStr safety and fail-fast validation."""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from rag.core.config import Settings


def _base_env(**overrides: str) -> dict[str, str]:
    """Return minimal valid environment for Settings."""
    env = {
        "COHERE_API_KEY": "test_key_0000000000000000000000",
        "JWT_SECRET_KEY": "test_jwt_secret_key_at_least_32_bytes!!",
    }
    env.update(overrides)
    return env


def test_settings_instantiate_with_valid_env() -> None:
    s = Settings(**_base_env())
    assert s.cohere_api_key is not None


def test_secret_str_not_in_repr() -> None:
    """SecretStr values must never appear in repr or string conversion."""
    s = Settings(**_base_env())
    repr_str = repr(s)
    assert "test_key" not in repr_str
    assert "test_jwt" not in repr_str


def test_secret_str_not_in_json() -> None:
    """SecretStr values must not appear in JSON serialisation."""
    s = Settings(**_base_env())
    data = s.model_dump()
    json_str = json.dumps(data)
    assert "test_key" not in json_str
    assert "test_jwt_secret_key" not in json_str


def test_secret_accessible_via_get_secret_value() -> None:
    """The secret value must be retrievable via get_secret_value()."""
    s = Settings(**_base_env())
    assert s.cohere_api_key.get_secret_value() == "test_key_0000000000000000000000"


def test_missing_cohere_api_key_raises() -> None:
    with pytest.raises(ValidationError, match="cohere_api_key"):
        Settings(jwt_secret_key="test_jwt_secret_key_at_least_32_bytes!!")


def test_missing_jwt_secret_raises() -> None:
    with pytest.raises(ValidationError, match="jwt_secret_key"):
        Settings(cohere_api_key="test_key")


def test_short_jwt_secret_raises() -> None:
    with pytest.raises(ValidationError, match="32"):
        Settings(**_base_env(JWT_SECRET_KEY="short"))


def test_hybrid_alpha_out_of_range_raises() -> None:
    with pytest.raises(ValidationError, match="HYBRID_ALPHA"):
        Settings(**_base_env(HYBRID_ALPHA="1.5"))

    with pytest.raises(ValidationError, match="HYBRID_ALPHA"):
        Settings(**_base_env(HYBRID_ALPHA="-0.1"))


def test_hybrid_alpha_boundary_values_valid() -> None:
    s0 = Settings(**_base_env(HYBRID_ALPHA="0.0"))
    assert s0.hybrid_alpha == 0.0

    s1 = Settings(**_base_env(HYBRID_ALPHA="1.0"))
    assert s1.hybrid_alpha == 1.0


def test_production_requires_qdrant_api_key() -> None:
    with pytest.raises(ValidationError, match="QDRANT_API_KEY"):
        Settings(**_base_env(APP_ENV="production"))


def test_production_with_qdrant_key_is_valid() -> None:
    s = Settings(**_base_env(APP_ENV="production", QDRANT_API_KEY="prod_qdrant_key"))
    assert s.app_env == "production"


def test_default_values_are_sane() -> None:
    s = Settings(**_base_env())
    assert s.retrieval_top_k == 50
    assert s.rerank_top_n == 10
    assert s.max_context_tokens == 4096
    assert s.log_level == "INFO"
    assert s.app_env == "development"
