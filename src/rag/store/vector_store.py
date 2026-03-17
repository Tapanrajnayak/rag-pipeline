"""Qdrant vector store — CRUD, collection lifecycle, ACL-filtered search."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from rag.core.errors import VectorStoreError
from rag.core.logging import get_logger
from rag.core.security import UserContext, build_acl_filter
from rag.ingestion.metadata import ChunkMetadata, DocumentMetadata

logger = get_logger(__name__)

# Cohere embed-v3 produces 1024-dimensional vectors
_EMBED_DIM = 1024
_SPARSE_VECTOR_NAME = "bm25"


class VectorStore:
    """Qdrant-backed vector store with ACL-filtered hybrid search.

    Collection schema:
    - Dense vectors: 1024-dim cosine similarity (Cohere embed-v3)
    - Sparse vectors: BM25 via fastembed (named 'bm25')
    - Payload: DocumentMetadata + ChunkMetadata, including acl_groups list
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    async def ensure_collection(self) -> None:
        """Create the collection if it does not already exist (idempotent).

        Raises:
            VectorStoreError: on unexpected Qdrant errors.
        """
        try:
            exists = await self._client.collection_exists(self._collection)
            if exists:
                logger.info("qdrant_collection_exists", collection=self._collection)
                return

            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=_EMBED_DIM,
                    distance=models.Distance.COSINE,
                    on_disk=False,
                ),
                sparse_vectors_config={
                    _SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            # Index the acl_groups payload field for fast filter queries
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name="acl_groups",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info("qdrant_collection_created", collection=self._collection)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialise Qdrant collection '{self._collection}'",
                detail=str(exc),
            ) from exc

    async def upsert_chunk(
        self,
        *,
        chunk_text: str,
        dense_vector: list[float],
        sparse_vector: dict[str, Any],  # {indices: list[int], values: list[float]}
        doc_meta: DocumentMetadata,
        chunk_meta: ChunkMetadata,
    ) -> str:
        """Upsert a single chunk into Qdrant.

        Args:
            chunk_text: The chunk text (stored in payload for retrieval).
            dense_vector: 1024-dim Cohere embed vector.
            sparse_vector: BM25 sparse vector dict with 'indices' and 'values'.
            doc_meta: Document-level metadata.
            chunk_meta: Chunk-level metadata.

        Returns:
            Point ID (UUID string).

        Raises:
            VectorStoreError: on Qdrant write failure.
        """
        point_id = str(uuid.uuid4())
        payload = chunk_meta.to_qdrant_payload(doc_meta)
        payload["text"] = chunk_text  # store text for retrieval

        try:
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "": dense_vector,  # default (unnamed) vector
                            _SPARSE_VECTOR_NAME: models.SparseVector(
                                indices=sparse_vector["indices"],
                                values=sparse_vector["values"],
                            ),
                        },
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert chunk for document '{doc_meta.document_id}'",
                detail=str(exc),
            ) from exc

        return point_id

    async def dense_search(
        self,
        query_vector: list[float],
        user: UserContext,
        *,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Semantic (dense) search with ACL filtering.

        Args:
            query_vector: Query embedding from Cohere.
            user: Authenticated user — ACL filter applied atomically.
            top_k: Number of results to return.

        Returns:
            List of result dicts with 'id', 'score', 'payload' keys.

        Raises:
            VectorStoreError: on Qdrant query failure.
        """
        acl_filter = build_acl_filter(user)
        try:
            results = await self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                query_filter=models.Filter(**_build_qdrant_filter(acl_filter)),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError("Dense search failed", detail=str(exc)) from exc

        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload or {}}
            for r in results
        ]

    async def sparse_search(
        self,
        sparse_vector: dict[str, Any],
        user: UserContext,
        *,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """BM25 (sparse) search with ACL filtering.

        Args:
            sparse_vector: BM25 sparse vector dict.
            user: Authenticated user — ACL filter applied atomically.
            top_k: Number of results to return.

        Returns:
            List of result dicts with 'id', 'score', 'payload' keys.

        Raises:
            VectorStoreError: on Qdrant query failure.
        """
        acl_filter = build_acl_filter(user)
        try:
            results = await self._client.search(
                collection_name=self._collection,
                query_vector=models.NamedSparseVector(
                    name=_SPARSE_VECTOR_NAME,
                    vector=models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                ),
                query_filter=models.Filter(**_build_qdrant_filter(acl_filter)),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError("Sparse search failed", detail=str(exc)) from exc

        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload or {}}
            for r in results
        ]

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document (GDPR right-to-erasure).

        Args:
            document_id: The document UUID to delete.

        Returns:
            Number of points deleted.

        Raises:
            VectorStoreError: on Qdrant delete failure.
        """
        try:
            result = await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            deleted = getattr(result, "result", None)
            count = deleted.deleted if deleted else 0
            logger.info("document_deleted", document_id=document_id, chunks_deleted=count)
            return count  # type: ignore[return-value]
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete document '{document_id}'",
                detail=str(exc),
            ) from exc

    async def health_check(self) -> bool:
        """Return True if Qdrant is reachable and the collection exists."""
        try:
            return await self._client.collection_exists(self._collection)
        except Exception:
            return False


def _build_qdrant_filter(acl_filter: dict[str, Any]) -> dict[str, Any]:
    """Convert the generic ACL filter dict to Qdrant Filter constructor kwargs.

    The ACL filter produced by build_acl_filter() uses a generic dict format.
    This function translates it into the kwargs expected by models.Filter().
    """
    must_conditions: list[models.Condition] = []
    for condition in acl_filter.get("must", []):
        key = condition["key"]
        match = condition["match"]
        if "any" in match:
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=match["any"]),
                )
            )
    return {"must": must_conditions}
