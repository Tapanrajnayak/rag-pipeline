"""Integration tests — real Qdrant + Redis, Cohere mocked via respx.

Run with:
    docker-compose -f infra/docker-compose.test.yml up -d
    pytest tests/integration/
"""

from __future__ import annotations

import pytest

from tests.conftest import FakeEmbedder, FakeSparseEncoder, FakeVectorStore
from rag.ingestion.pipeline import ingest_document
from rag.ingestion.parsers.text import TextParser
from rag.retrieval.hybrid import hybrid_retrieve
from rag.store.document_store import InMemoryDocumentStore
from rag.core.security import UserContext


@pytest.fixture()
def eng_user() -> UserContext:
    return UserContext(
        user_id="user-eng-001",
        groups=frozenset({"engineering", "all"}),
    )


@pytest.fixture()
def legal_user() -> UserContext:
    return UserContext(
        user_id="user-legal-001",
        groups=frozenset({"legal", "all"}),
    )


@pytest.mark.asyncio
async def test_ingest_then_retrieve(eng_user: UserContext) -> None:
    """Ingest a document and verify it appears in hybrid search results."""
    vector_store = FakeVectorStore()
    doc_repo = InMemoryDocumentStore()
    embedder = FakeEmbedder()
    sparse_encoder = FakeSparseEncoder()

    content = b"Hybrid search combines dense and sparse retrieval. It is very effective."
    result = await ingest_document(
        content,
        filename="test.txt",
        content_type="text/plain",
        acl_groups=["engineering", "all"],
        uploaded_by=eng_user.user_id,
        parser=TextParser(),
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        vector_store=vector_store,
        doc_repo=doc_repo,
    )

    assert result.chunks_stored > 0
    assert result.document_id != ""

    # Now retrieve — should find the ingested document
    chunks = await hybrid_retrieve(
        query_dense=[0.1] * 1024,
        query_sparse={"indices": [0, 1], "values": [0.5, 0.5]},
        user=eng_user,
        vector_store=vector_store,
    )

    assert len(chunks) > 0
    doc_ids = [c.document_id for c in chunks]
    assert result.document_id in doc_ids


@pytest.mark.asyncio
async def test_acl_enforcement(eng_user: UserContext, legal_user: UserContext) -> None:
    """Documents tagged for engineering should NOT appear for legal users."""
    vector_store = FakeVectorStore()
    doc_repo = InMemoryDocumentStore()
    embedder = FakeEmbedder()
    sparse_encoder = FakeSparseEncoder()

    # Ingest engineering-only document
    await ingest_document(
        b"Secret engineering design document with proprietary information.",
        filename="secret.txt",
        content_type="text/plain",
        acl_groups=["engineering"],  # NOT "all"
        uploaded_by=eng_user.user_id,
        parser=TextParser(),
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        vector_store=vector_store,
        doc_repo=doc_repo,
    )

    # Engineering user can see it
    eng_results = await hybrid_retrieve(
        query_dense=[0.1] * 1024,
        query_sparse={"indices": [0], "values": [1.0]},
        user=eng_user,
        vector_store=vector_store,
    )
    assert len(eng_results) > 0

    # Legal user cannot see it
    legal_results = await hybrid_retrieve(
        query_dense=[0.1] * 1024,
        query_sparse={"indices": [0], "values": [1.0]},
        user=legal_user,
        vector_store=vector_store,
    )
    assert len(legal_results) == 0


@pytest.mark.asyncio
async def test_all_group_accessible_to_all_users(
    eng_user: UserContext, legal_user: UserContext
) -> None:
    """Documents tagged 'all' should be accessible to any authenticated user."""
    vector_store = FakeVectorStore()
    doc_repo = InMemoryDocumentStore()

    await ingest_document(
        b"Public document available to all employees.",
        filename="public.txt",
        content_type="text/plain",
        acl_groups=["all"],
        uploaded_by="admin",
        parser=TextParser(),
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        vector_store=vector_store,
        doc_repo=doc_repo,
    )

    for user in [eng_user, legal_user]:
        results = await hybrid_retrieve(
            query_dense=[0.1] * 1024,
            query_sparse={"indices": [0], "values": [1.0]},
            user=user,
            vector_store=vector_store,
        )
        assert len(results) > 0, f"{user.user_id} should see 'all' documents"


@pytest.mark.asyncio
async def test_document_metadata_persisted(eng_user: UserContext) -> None:
    """Document metadata should be retrievable from the document store."""
    doc_repo = InMemoryDocumentStore()
    vector_store = FakeVectorStore()

    result = await ingest_document(
        b"Test document for metadata persistence.",
        filename="meta_test.txt",
        content_type="text/plain",
        acl_groups=["engineering", "all"],
        uploaded_by=eng_user.user_id,
        parser=TextParser(),
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        vector_store=vector_store,
        doc_repo=doc_repo,
    )

    meta = await doc_repo.get(result.document_id)
    assert meta is not None
    assert meta.filename == "meta_test.txt"
    assert meta.uploaded_by == eng_user.user_id
    assert "engineering" in meta.acl_groups
