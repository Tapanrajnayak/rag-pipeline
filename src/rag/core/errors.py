"""Typed exception hierarchy — every error has an HTTP status and error code."""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""

    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ── Auth errors ───────────────────────────────────────────────────────────────


class AuthenticationError(RAGError):
    http_status = 401
    error_code = "authentication_error"


class AuthorizationError(RAGError):
    http_status = 403
    error_code = "authorization_error"


# ── Client errors ─────────────────────────────────────────────────────────────


class ValidationError(RAGError):
    http_status = 422
    error_code = "validation_error"


class NotFoundError(RAGError):
    http_status = 404
    error_code = "not_found"


class RateLimitError(RAGError):
    http_status = 429
    error_code = "rate_limit_exceeded"


# ── Upstream / infra errors ───────────────────────────────────────────────────


class EmbeddingError(RAGError):
    http_status = 502
    error_code = "embedding_error"


class RetrievalError(RAGError):
    http_status = 502
    error_code = "retrieval_error"


class GenerationError(RAGError):
    http_status = 502
    error_code = "generation_error"


class VectorStoreError(RAGError):
    http_status = 502
    error_code = "vector_store_error"


class DocumentParseError(RAGError):
    http_status = 422
    error_code = "document_parse_error"
