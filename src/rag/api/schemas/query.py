"""Request/response schemas for the query API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request body."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Number of candidate chunks to retrieve before reranking.",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply Cohere Rerank v3.",
    )


class Citation(BaseModel):
    """A source citation in the query response."""

    index: int
    document_id: str
    chunk_id: str
    text: str


class QueryResponse(BaseModel):
    """Query response envelope.

    `model_versions` is always included — regulated-industry audit requirement:
    auditors must be able to determine which model versions produced an answer.
    """

    data: str                          # the generated answer
    citations: list[Citation]
    request_id: str
    model_versions: dict[str, str]     # e.g. {"embed": "...", "rerank": "...", "generate": "..."}
    latency_ms: float
    rerank_used: bool
    context_tokens: int
