"""Query API — POST /v1/query."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from rag.api.dependencies import (
    get_current_user,
    get_embedder,
    get_sparse_encoder,
    get_vector_store,
)
from rag.api.schemas.query import Citation, QueryRequest, QueryResponse
from rag.core.config import Settings, get_settings
from rag.core.security import UserContext
from rag.embedding.cache import EmbeddingCache
from rag.generation.context import pack_context
from rag.generation.generator import generate
from rag.observability.audit import log_query_event
from rag.observability.metrics import (
    EMBEDDING_CACHE_HITS,
    EMBEDDING_CACHE_MISSES,
    QUERY_DURATION,
    QUERY_TOTAL,
)
from rag.retrieval.hybrid import hybrid_retrieve
from rag.retrieval.reranker import rerank

router = APIRouter(prefix="/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline",
)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    vector_store=Depends(get_vector_store),  # type: ignore[no-untyped-def]
    embedder=Depends(get_embedder),  # type: ignore[no-untyped-def]
    sparse_encoder=Depends(get_sparse_encoder),  # type: ignore[no-untyped-def]
) -> QueryResponse:
    """Execute a RAG query.

    Pipeline:
    1. Embed query (cache-backed)
    2. BM25 sparse encode query
    3. Hybrid retrieve (parallel dense + sparse, RRF fusion) — ACL enforced
    4. Rerank with Cohere Rerank v3 (circuit breaker fallback)
    5. Pack context within token budget
    6. Generate grounded answer with Command R+
    7. Return response with model_versions + citations
    """
    start = time.monotonic()
    request_id = getattr(request.state, "request_id", "")
    api_key = settings.cohere_api_key.get_secret_value()

    # ── 1. Embed query (with Redis cache) ────────────────────────────────────
    embedding_cache: EmbeddingCache | None = getattr(
        request.app.state, "embedding_cache", None
    )
    query_dense = None
    if embedding_cache:
        query_dense = await embedding_cache.get(body.query)

    if query_dense is None:
        EMBEDDING_CACHE_MISSES.inc()
        query_dense = await embedder.embed_query(body.query)
        if embedding_cache:
            await embedding_cache.set(body.query, query_dense)
    else:
        EMBEDDING_CACHE_HITS.inc()

    # ── 2. BM25 sparse encode ─────────────────────────────────────────────────
    query_sparse = await sparse_encoder.encode(body.query)

    # ── 3. Hybrid retrieve (ACL enforced inside) ──────────────────────────────
    candidates = await hybrid_retrieve(
        query_dense=query_dense,
        query_sparse=query_sparse,
        user=user,
        vector_store=vector_store,
        top_k=body.top_k,
        alpha=settings.hybrid_alpha,
    )

    if not candidates:
        # No documents found for this user's ACL — return empty, not 403
        # (leaking existence of documents is a security issue)
        latency_ms = (time.monotonic() - start) * 1000
        QUERY_TOTAL.labels(status="empty").inc()
        return QueryResponse(
            data="I don't have any relevant documents to answer your question.",
            citations=[],
            request_id=request_id,
            model_versions=_model_versions(settings, rerank_used=False),
            latency_ms=round(latency_ms, 2),
            rerank_used=False,
            context_tokens=0,
        )

    # ── 4. Rerank ─────────────────────────────────────────────────────────────
    rerank_used = body.rerank
    reranked = await rerank(
        body.query,
        candidates,
        api_key=api_key,
        model=settings.cohere_rerank_model,
        top_n=settings.rerank_top_n,
    )
    actual_rerank_used = reranked[0].rerank_used if reranked else False

    # ── 5. Pack context ───────────────────────────────────────────────────────
    context = pack_context(reranked, max_tokens=settings.max_context_tokens)

    # ── 6. Generate ───────────────────────────────────────────────────────────
    try:
        generation = await generate(
            body.query,
            context,
            api_key=api_key,
            model=settings.cohere_generate_model,
        )
    except Exception as exc:
        QUERY_TOTAL.labels(status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Generation failed: {exc}",
        ) from exc

    latency_ms = (time.monotonic() - start) * 1000
    model_versions = _model_versions(settings, rerank_used=actual_rerank_used)

    # ── 7. Record metrics and audit ───────────────────────────────────────────
    QUERY_TOTAL.labels(status="success").inc()
    QUERY_DURATION.labels(
        status="success",
        rerank_used=str(actual_rerank_used).lower(),
    ).observe(latency_ms / 1000)

    log_query_event(
        body.query,
        user_id=user.user_id,
        document_ids=[c.document_id for c in context.citations],
        model_versions=model_versions,
        latency_ms=latency_ms,
        rerank_used=actual_rerank_used,
        chunk_count=len(context.citations),
        request_id=request_id,
    )

    return QueryResponse(
        data=generation.answer,
        citations=[
            Citation(
                index=c.index,
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                text=c.text,
            )
            for c in context.citations
        ],
        request_id=request_id,
        model_versions=model_versions,
        latency_ms=round(latency_ms, 2),
        rerank_used=actual_rerank_used,
        context_tokens=context.token_count,
    )


def _model_versions(settings: Settings, *, rerank_used: bool) -> dict[str, str]:
    """Build the model_versions dict for the response envelope."""
    versions = {
        "embed": settings.cohere_embed_model,
        "generate": settings.cohere_generate_model,
    }
    if rerank_used:
        versions["rerank"] = settings.cohere_rerank_model
    return versions
