"""FastAPI application factory with lifespan context manager."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from qdrant_client import AsyncQdrantClient

from rag.api.middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from rag.api.routers import documents, health, query
from rag.core.config import Settings, get_settings
from rag.core.errors import RAGError
from rag.core.logging import configure_logging, get_logger
from rag.embedding.cache import EmbeddingCache
from rag.embedding.cohere import CohereEmbeddingProvider
from rag.retrieval.sparse import SparseEncoder
from rag.store.document_store import InMemoryDocumentStore
from rag.store.vector_store import VectorStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and graceful shutdown.

    Startup:
    - Validate configuration (fails fast)
    - Initialise Qdrant collection (idempotent)
    - Warm Redis connection pool
    - Attach services to app.state for DI

    Shutdown:
    - Close Qdrant and Redis connections gracefully
    """
    settings: Settings = get_settings()

    configure_logging(
        level=settings.log_level,
        json_output=settings.app_env == "production",
    )

    logger.info("startup_begin", env=settings.app_env)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_kwargs: dict[str, object] = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        qdrant_kwargs["api_key"] = settings.qdrant_api_key.get_secret_value()

    qdrant_client = AsyncQdrantClient(**qdrant_kwargs)
    vector_store = VectorStore(qdrant_client, settings.qdrant_collection)
    await vector_store.ensure_collection()

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_client: aioredis.Redis = aioredis.from_url(  # type: ignore[type-arg]
        settings.redis_url,
        decode_responses=False,
        socket_connect_timeout=5,
    )
    await redis_client.ping()  # fail fast on misconfigured Redis

    embedding_cache = EmbeddingCache(
        redis_client,
        model=settings.cohere_embed_model,
    )

    # ── Cohere / sparse encoder ───────────────────────────────────────────────
    embedder = CohereEmbeddingProvider(
        api_key=settings.cohere_api_key.get_secret_value(),
        model=settings.cohere_embed_model,
    )
    sparse_encoder = SparseEncoder()

    # ── Document repo (in-memory for demo; swap to Postgres for prod) ─────────
    doc_repo = InMemoryDocumentStore()

    # Attach to app state for DI
    app.state.vector_store = vector_store
    app.state.redis_client = redis_client
    app.state.embedding_cache = embedding_cache
    app.state.embedder = embedder
    app.state.sparse_encoder = sparse_encoder
    app.state.doc_repo = doc_repo

    logger.info("startup_complete")

    yield  # ── Application running ──────────────────────────────────────────

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    logger.info("shutdown_begin")
    await redis_client.aclose()
    await qdrant_client.close()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Application factory — returns a fully configured FastAPI instance.

    Using a factory function (rather than a module-level app) enables:
    - Clean test setup: each test gets a fresh app with DI overrides
    - Multiple app variants (e.g., with/without auth) without global state
    """
    settings = get_settings()

    app = FastAPI(
        title="RAG Pipeline",
        description=(
            "Production hybrid-search RAG pipeline — Cohere embed/rerank/generate, "
            "Qdrant dense+sparse, ACL enforcement, audit logging."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan,
    )

    # ── Middleware (outermost first) ───────────────────────────────────────────
    # Note: Starlette applies middleware in reverse registration order,
    # so the last-added middleware runs first.
    # We register in the order we want them to execute:
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    # Rate limit middleware is added after app state is populated in lifespan,
    # but Starlette requires middleware registration before startup.
    # We attach Redis lazily via app.state in the middleware.

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)

    # ── Prometheus metrics endpoint ───────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(RAGError)
    async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "detail": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    return app


# Module-level app for uvicorn entrypoint
app = create_app()
