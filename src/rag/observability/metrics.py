"""Prometheus metrics — counters, histograms, gauges."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

# ── Query metrics ─────────────────────────────────────────────────────────────

QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "End-to-end query latency",
    labelnames=["status", "rerank_used"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total number of queries",
    labelnames=["status"],
)

# ── Retrieval metrics ─────────────────────────────────────────────────────────

RETRIEVAL_RESULTS_TOTAL = Counter(
    "rag_retrieval_results_total",
    "Total chunks retrieved, by search modality",
    labelnames=["modality"],  # dense | sparse | hybrid
)

RERANK_FALLBACK_TOTAL = Counter(
    "rag_rerank_fallback_total",
    "Number of times reranker circuit breaker activated",
)

# ── Embedding cache ───────────────────────────────────────────────────────────

EMBEDDING_CACHE_HITS = Counter(
    "rag_embedding_cache_hits_total",
    "Embedding cache hits",
)

EMBEDDING_CACHE_MISSES = Counter(
    "rag_embedding_cache_misses_total",
    "Embedding cache misses",
)

# ── Document metrics ──────────────────────────────────────────────────────────

DOCUMENT_COUNT = Gauge(
    "rag_document_count",
    "Number of documents in the store",
    labelnames=["acl_group"],
)

# ── Ingestion metrics ─────────────────────────────────────────────────────────

INGEST_DURATION = Histogram(
    "rag_ingest_duration_seconds",
    "Document ingestion latency",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0],
)

INGEST_TOTAL = Counter(
    "rag_ingest_total",
    "Total ingestion operations",
    labelnames=["status"],
)
