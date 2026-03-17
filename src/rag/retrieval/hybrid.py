"""Async parallel hybrid retrieval — dense + sparse — fused with RRF.

Reciprocal Rank Fusion (RRF):
    score(d) = Σ_i  1 / (k + rank_i(d))

where k=60 is the standard smoothing constant from Cormack et al. (2009).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from rag.core.logging import get_logger
from rag.core.security import UserContext
from rag.store.vector_store import VectorStore

logger = get_logger(__name__)

_RRF_K = 60  # standard RRF smoothing constant


@dataclass
class RetrievedChunk:
    """A single chunk returned by retrieval, with all context for reranking."""

    chunk_id: str
    text: str
    document_id: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


async def hybrid_retrieve(
    query_dense: list[float],
    query_sparse: dict[str, Any],
    user: UserContext,
    vector_store: VectorStore,
    *,
    top_k: int = 50,
    alpha: float = 0.5,  # kept for documentation; RRF does not use it directly
) -> list[RetrievedChunk]:
    """Run dense and sparse search in parallel, fuse results with RRF.

    Both searches are launched with asyncio.gather so the network round-trips
    overlap. ACL filtering is applied atomically inside each search call.

    Args:
        query_dense: Dense query embedding vector.
        query_sparse: BM25 sparse vector dict.
        user: Authenticated user (groups propagated to ACL filter).
        vector_store: Qdrant store abstraction.
        top_k: Number of candidates to retrieve from each modality.
        alpha: Kept for API compatibility and docs; RRF is rank-based, not
               score-based, so alpha does not change the formula.

    Returns:
        List of RetrievedChunk sorted by RRF score descending.
    """
    dense_results, sparse_results = await asyncio.gather(
        vector_store.dense_search(query_dense, user, top_k=top_k),
        vector_store.sparse_search(query_sparse, user, top_k=top_k),
    )

    logger.debug(
        "hybrid_raw_counts",
        dense=len(dense_results),
        sparse=len(sparse_results),
    )

    fused = _rrf_fuse(dense_results, sparse_results)

    logger.info(
        "hybrid_retrieve_ok",
        fused_count=len(fused),
        top_k=top_k,
        user_id=user.user_id,
    )
    return fused


def _rrf_fuse(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    *,
    k: int = _RRF_K,
) -> list[RetrievedChunk]:
    """Apply Reciprocal Rank Fusion to two ranked result lists.

    Args:
        dense: Dense search results (ordered by score descending).
        sparse: Sparse search results (ordered by score descending).
        k: RRF smoothing constant (default: 60).

    Returns:
        Merged list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(dense):
        chunk_id = hit["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        dense_ranks[chunk_id] = rank
        payloads[chunk_id] = hit["payload"]

    for rank, hit in enumerate(sparse):
        chunk_id = hit["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        sparse_ranks[chunk_id] = rank
        if chunk_id not in payloads:
            payloads[chunk_id] = hit["payload"]

    chunks: list[RetrievedChunk] = []
    for chunk_id, rrf_score in scores.items():
        payload = payloads[chunk_id]
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=payload.get("text", ""),
                document_id=payload.get("document_id", ""),
                score=rrf_score,
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
                rrf_score=rrf_score,
                payload=payload,
            )
        )

    # Sort by RRF score descending; ties broken by chunk_id for determinism
    chunks.sort(key=lambda c: (-c.rrf_score, c.chunk_id))
    return chunks
