"""Test fixtures — DI overrides, fakes, and shared test utilities."""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# ── Environment setup (must happen before any rag imports) ────────────────────
os.environ.setdefault("COHERE_API_KEY", "test_cohere_key_00000000000000000000")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_at_least_32_bytes!!")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from rag.core.config import Settings
from rag.core.security import UserContext, create_access_token
from rag.ingestion.chunker import Chunk
from rag.retrieval.hybrid import RetrievedChunk
from rag.retrieval.reranker import RerankResult
from rag.store.document_store import InMemoryDocumentStore


# ── Settings fixture ──────────────────────────────────────────────────────────

@pytest.fixture()
def settings() -> Settings:
    """Return a Settings instance with test values."""
    return Settings(
        cohere_api_key="test_cohere_key_00000000000000000000",
        jwt_secret_key="test_jwt_secret_key_at_least_32_bytes!!",
    )


# ── User fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def engineering_user() -> UserContext:
    return UserContext(
        user_id="user-eng-001",
        groups=frozenset({"engineering", "all"}),
        email="eng@example.com",
    )


@pytest.fixture()
def legal_user() -> UserContext:
    return UserContext(
        user_id="user-legal-001",
        groups=frozenset({"legal", "all"}),
        email="legal@example.com",
    )


@pytest.fixture()
def admin_user() -> UserContext:
    return UserContext(
        user_id="user-admin-001",
        groups=frozenset({"admin", "all"}),
        email="admin@example.com",
    )


@pytest.fixture()
def engineering_token(settings: Settings) -> str:
    return create_access_token(
        user_id="user-eng-001",
        groups=["engineering", "all"],
        secret_key=settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
    )


@pytest.fixture()
def legal_token(settings: Settings) -> str:
    return create_access_token(
        user_id="user-legal-001",
        groups=["legal", "all"],
        secret_key=settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
    )


# ── Fake services (no I/O) ────────────────────────────────────────────────────

class FakeEmbedder:
    """Returns deterministic unit vectors — no Cohere API calls."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * 1024


class FakeSparseEncoder:
    """Returns a trivial sparse vector — no fastembed model loading."""

    async def encode(self, text: str) -> dict[str, Any]:
        return {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}

    async def encode_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        return [{"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]} for _ in texts]


class FakeVectorStore:
    """In-memory vector store — no Qdrant."""

    def __init__(self) -> None:
        self._points: list[dict[str, Any]] = []
        self._document_ids: set[str] = set()

    async def ensure_collection(self) -> None:
        pass

    async def upsert_chunk(self, **kwargs: Any) -> str:
        import uuid
        point_id = str(uuid.uuid4())
        self._points.append(
            {
                "id": point_id,
                "text": kwargs.get("chunk_text", ""),
                "document_id": kwargs["doc_meta"].document_id,
                "acl_groups": kwargs["doc_meta"].acl_groups,
                "payload": kwargs["chunk_meta"].to_qdrant_payload(kwargs["doc_meta"]),
            }
        )
        self._document_ids.add(kwargs["doc_meta"].document_id)
        return point_id

    async def dense_search(
        self, query_vector: list[float], user: UserContext, *, top_k: int = 50
    ) -> list[dict[str, Any]]:
        results = []
        for i, p in enumerate(self._points):
            if any(g in user.acl_groups for g in p["acl_groups"]):
                results.append({
                    "id": p["id"],
                    "score": 1.0 - i * 0.01,
                    "payload": {**p["payload"], "text": p["text"]},
                })
        return results[:top_k]

    async def sparse_search(
        self, sparse_vector: dict[str, Any], user: UserContext, *, top_k: int = 50
    ) -> list[dict[str, Any]]:
        return await self.dense_search([0.1] * 1024, user, top_k=top_k)

    async def delete_document(self, document_id: str) -> int:
        original = len(self._points)
        self._points = [p for p in self._points if p["document_id"] != document_id]
        self._document_ids.discard(document_id)
        return original - len(self._points)

    async def health_check(self) -> bool:
        return True


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def fake_sparse_encoder() -> FakeSparseEncoder:
    return FakeSparseEncoder()


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def fake_doc_repo() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


# ── Sample data fixtures ──────────────────────────────────────────────────────

@pytest.fixture()
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-001",
        text="Hybrid search combines dense and sparse retrieval for better results.",
        document_id="doc-001",
        score=0.85,
        dense_rank=0,
        sparse_rank=1,
        rrf_score=0.85,
        payload={
            "document_id": "doc-001",
            "acl_groups": ["engineering", "all"],
        },
    )


@pytest.fixture()
def sample_rerank_result(sample_chunk: RetrievedChunk) -> RerankResult:
    return RerankResult(
        chunk=sample_chunk,
        relevance_score=0.92,
        rerank_used=True,
    )
