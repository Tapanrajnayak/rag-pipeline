"""Cohere Rerank v3 with circuit breaker fallback.

Circuit breaker behaviour:
- On HTTP 503 / timeout / any Cohere API error: log metric, return original
  hybrid-ranked order (degraded quality but query never fails).
- No state machine; this is a simple per-request try/except pattern.
  A production system would add a stateful breaker (half-open state, etc.)
  via a library like `circuitbreaker` or Resilience4j.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cohere

from rag.core.errors import RetrievalError
from rag.core.logging import get_logger
from rag.observability.metrics import RERANK_FALLBACK_TOTAL
from rag.retrieval.hybrid import RetrievedChunk

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Reranked chunk with Cohere relevance score."""

    chunk: RetrievedChunk
    relevance_score: float
    rerank_used: bool  # False if circuit breaker activated


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    api_key: str,
    model: str = "rerank-english-v3.0",
    top_n: int = 10,
) -> list[RerankResult]:
    """Rerank chunks with Cohere Rerank v3.

    Retrieves top-50 from hybrid, reranks to top_n. On any API failure,
    falls back to the original RRF ordering (circuit breaker).

    Args:
        query: Original user query string.
        chunks: Candidates from hybrid retrieval (up to 50).
        api_key: Cohere API key.
        model: Cohere rerank model identifier.
        top_n: Number of results to return after reranking.

    Returns:
        List of RerankResult sorted by relevance score descending.
        If circuit breaker fires, `rerank_used=False` on all results.
    """
    if not chunks:
        return []

    try:
        results = await _call_rerank(query, chunks, api_key=api_key, model=model, top_n=top_n)
        logger.info(
            "rerank_ok",
            input_count=len(chunks),
            output_count=len(results),
            model=model,
        )
        return results
    except Exception as exc:
        # Circuit breaker: degrade gracefully, never propagate failure
        RERANK_FALLBACK_TOTAL.inc()
        logger.warning(
            "rerank_circuit_breaker",
            error=str(exc),
            fallback="rrf_order",
        )
        # Return top_n from hybrid order with rerank_used=False
        return [
            RerankResult(
                chunk=c,
                relevance_score=c.rrf_score,
                rerank_used=False,
            )
            for c in chunks[:top_n]
        ]


async def _call_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    api_key: str,
    model: str,
    top_n: int,
) -> list[RerankResult]:
    """Internal: call Cohere Rerank API in thread pool.

    Args:
        query: User query.
        chunks: Candidate chunks.
        api_key: Cohere API key.
        model: Rerank model.
        top_n: Top N to keep.

    Returns:
        Reranked results.

    Raises:
        Exception: any Cohere API error (caller handles circuit breaking).
    """
    client = cohere.Client(api_key=api_key)
    documents = [c.text for c in chunks]

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n,
            return_documents=False,
        ),
    )

    results: list[RerankResult] = []
    for item in response.results:
        chunk = chunks[item.index]
        results.append(
            RerankResult(
                chunk=chunk,
                relevance_score=item.relevance_score,
                rerank_used=True,
            )
        )
    return results
