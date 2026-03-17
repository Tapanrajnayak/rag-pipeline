#!/usr/bin/env python3
"""Seed the demo corpus with sample documents across 3 ACL groups.

Usage:
    uv run python scripts/seed_demo.py

Requires:
    - Running Qdrant + Redis (docker-compose up -d)
    - COHERE_API_KEY set in environment
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.core.config import get_settings
from rag.embedding.cohere import CohereEmbeddingProvider
from rag.ingestion.parsers.text import TextParser
from rag.ingestion.pipeline import ingest_document
from rag.retrieval.sparse import SparseEncoder
from rag.store.document_store import InMemoryDocumentStore
from rag.store.vector_store import VectorStore

from qdrant_client import AsyncQdrantClient

# ── Sample documents ──────────────────────────────────────────────────────────
# In a real demo: download from arXiv API and parse PDFs.
# Here: curated text excerpts covering the RAG domain.

DOCUMENTS = [
    {
        "title": "Dense Passage Retrieval for Open-Domain QA",
        "text": (
            "Dense passage retrieval (DPR) uses dual-encoder neural networks to embed "
            "both questions and passages into a shared dense vector space. Retrieval is "
            "performed via maximum inner product search, enabling efficient semantic matching. "
            "DPR significantly outperforms BM25 on open-domain question answering benchmarks "
            "such as Natural Questions and TriviaQA, especially for paraphrase and synonym "
            "heavy queries where lexical matching fails."
        ),
        "acl_groups": ["engineering", "all"],
    },
    {
        "title": "Improving Retrieval with Hybrid Search",
        "text": (
            "Hybrid search combines sparse lexical retrieval (BM25) with dense semantic "
            "retrieval to leverage complementary strengths. Sparse methods excel at exact "
            "keyword matching and rare term recall. Dense methods generalise to synonyms "
            "and paraphrases. Reciprocal Rank Fusion (RRF) is a score-free fusion method "
            "that combines ranked lists without requiring calibrated scores. RRF with k=60 "
            "consistently outperforms individual retrieval modalities across multiple benchmarks."
        ),
        "acl_groups": ["engineering", "all"],
    },
    {
        "title": "Cohere Rerank: Cross-Encoder Reranking",
        "text": (
            "Cross-encoder reranking models jointly encode the query and each candidate "
            "passage, enabling fine-grained relevance assessment. Unlike bi-encoders used "
            "in retrieval, cross-encoders attend across the full query-document pair. "
            "Cohere Rerank v3 uses this architecture to reorder top-k retrieved candidates, "
            "significantly improving precision at the cost of additional latency. "
            "The typical pattern is: retrieve 50 candidates, rerank to top 10."
        ),
        "acl_groups": ["engineering", "all"],
    },
    {
        "title": "RAG Security and Privacy Considerations",
        "text": (
            "Retrieval-Augmented Generation systems in regulated industries must implement "
            "strict access controls. Access Control Lists (ACLs) at the vector store layer "
            "ensure that users can only retrieve documents they are authorised to access. "
            "Audit logs must capture query metadata without storing raw query text, which "
            "may contain personally identifiable information (PII) such as patient names "
            "or account numbers. Query hashing (sha256) allows security teams to correlate "
            "events without exposing sensitive content."
        ),
        "acl_groups": ["legal", "all"],
    },
    {
        "title": "GDPR Right to Erasure in Document Systems",
        "text": (
            "The GDPR right to erasure (Article 17) requires that personal data be deleted "
            "upon request within 30 days. For RAG systems, this means deleting all vector "
            "embeddings, document chunks, and associated metadata for the requested document. "
            "Qdrant supports payload-filtered deletion, enabling efficient erasure by "
            "document ID without rebuilding the entire collection. Audit logs of the erasure "
            "operation must be retained separately for compliance purposes."
        ),
        "acl_groups": ["legal", "all"],
    },
    {
        "title": "Cohere Command R+: Grounded Generation",
        "text": (
            "Cohere Command R+ is optimised for Retrieval-Augmented Generation with native "
            "support for the documents= parameter. When documents are provided, the model "
            "grounds its answers in those sources and returns structured citation objects "
            "linking spans of the answer to specific document segments. This enables "
            "verifiable, auditable responses suitable for regulated industries. "
            "Temperature=0 produces deterministic outputs for reproducible evaluations."
        ),
        "acl_groups": ["all"],
    },
]


async def main() -> None:
    settings = get_settings()

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    vector_store = VectorStore(qdrant_client, settings.qdrant_collection)
    await vector_store.ensure_collection()

    embedder = CohereEmbeddingProvider(
        api_key=settings.cohere_api_key.get_secret_value(),
        model=settings.cohere_embed_model,
    )
    sparse_encoder = SparseEncoder()
    doc_repo = InMemoryDocumentStore()
    parser = TextParser()

    print(f"Seeding {len(DOCUMENTS)} documents...\n")

    for doc in DOCUMENTS:
        content = doc["text"].encode("utf-8")
        result = await ingest_document(
            content,
            filename=f"{doc['title'].lower().replace(' ', '_')}.txt",
            content_type="text/plain",
            acl_groups=doc["acl_groups"],
            uploaded_by="seed_script",
            parser=parser,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            vector_store=vector_store,
            doc_repo=doc_repo,
        )
        groups_str = ", ".join(doc["acl_groups"])
        print(f"  ✓ [{groups_str}] {doc['title']} → {result.chunks_stored} chunk(s)")

    print(f"\nDone. {len(DOCUMENTS)} documents ingested.")
    await qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(main())
