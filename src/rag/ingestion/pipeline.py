"""Ingestion pipeline — parse → chunk → embed → sparse encode → store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from rag.core.logging import get_logger
from rag.embedding.base import EmbeddingProvider
from rag.ingestion.chunker import chunk_text
from rag.ingestion.metadata import ChunkMetadata, DocumentMetadata
from rag.ingestion.parsers.base import Parser
from rag.retrieval.sparse import SparseEncoder
from rag.store.document_store import DocumentRepository
from rag.store.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting a single document."""

    document_id: str
    chunks_stored: int
    filename: str


async def ingest_document(
    content: bytes,
    *,
    filename: str,
    content_type: str,
    acl_groups: list[str],
    uploaded_by: str,
    parser: Parser,
    embedder: EmbeddingProvider,
    sparse_encoder: SparseEncoder,
    vector_store: VectorStore,
    doc_repo: DocumentRepository,
    chunk_max_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> IngestResult:
    """Run the full ingestion pipeline for a single document.

    Steps:
    1. Parse raw bytes → plain text
    2. Chunk text with sentence-boundary sliding window
    3. Batch-embed all chunks (search_document input_type)
    4. BM25 sparse encode all chunks
    5. Upsert each chunk to Qdrant (dense + sparse + payload)
    6. Persist document metadata to document store

    Args:
        content: Raw file bytes.
        filename: Original filename.
        content_type: MIME type string.
        acl_groups: ACL groups that may access this document.
        uploaded_by: User ID of the uploader.
        parser: Document parser (PDF, DOCX, text, etc.).
        embedder: Embedding provider (Cohere).
        sparse_encoder: BM25 sparse encoder.
        vector_store: Qdrant vector store.
        doc_repo: Document metadata store.
        chunk_max_tokens: Max tokens per chunk.
        chunk_overlap_tokens: Overlap tokens between chunks.

    Returns:
        IngestResult with document ID and chunk count.
    """
    document_id = str(uuid.uuid4())
    logger.info("ingest_start", document_id=document_id, filename=filename)

    # Step 1: Parse
    text = parser.parse(content, filename=filename)
    logger.debug("parse_ok", document_id=document_id, text_len=len(text))

    # Step 2: Chunk
    chunks = chunk_text(
        text,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
    )
    if not chunks:
        logger.warning("no_chunks", document_id=document_id, filename=filename)
        return IngestResult(document_id=document_id, chunks_stored=0, filename=filename)

    logger.debug("chunk_ok", document_id=document_id, chunk_count=len(chunks))

    # Step 3: Dense embed all chunks (batched)
    texts = [c.text for c in chunks]
    dense_vectors = await embedder.embed_documents(texts)

    # Step 4: Sparse encode all chunks
    sparse_vectors = await sparse_encoder.encode_batch(texts)

    # Step 5: Build metadata
    doc_meta = DocumentMetadata(
        document_id=document_id,
        filename=filename,
        content_type=content_type,
        acl_groups=acl_groups,
        uploaded_by=uploaded_by,
    )

    # Step 6: Upsert chunks to Qdrant
    for i, (chunk, dense_vec, sparse_vec) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors, strict=True)
    ):
        chunk_meta = ChunkMetadata(
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            start_sentence=chunk.start_sentence,
            end_sentence=chunk.end_sentence,
            token_count=chunk.token_count,
            acl_groups=acl_groups,
        )
        await vector_store.upsert_chunk(
            chunk_text=chunk.text,
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            doc_meta=doc_meta,
            chunk_meta=chunk_meta,
        )

    # Step 7: Persist document metadata
    await doc_repo.save(doc_meta)

    logger.info(
        "ingest_ok",
        document_id=document_id,
        filename=filename,
        chunks=len(chunks),
    )
    return IngestResult(
        document_id=document_id,
        chunks_stored=len(chunks),
        filename=filename,
    )
