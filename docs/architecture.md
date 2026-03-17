# Architecture

## System Diagram

```
Client
  │
  │  HTTP (Bearer JWT)
  ▼
┌──────────────────────────────────────────────────────┐
│  FastAPI  (src/rag/api/)                             │
│  ┌──────────────────────────────────────────────┐   │
│  │  Middleware chain (in execution order)       │   │
│  │  1. SecurityHeaders                          │   │
│  │  2. RequestID injection                      │   │
│  │  3. Structured request/response logging      │   │
│  │  4. Redis sliding-window rate limiter        │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ /v1/docs │  │/v1/query │  │ /health/live|ready│  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────┘
         │                 │
  [Ingest path]     [Query path]
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────────────────────┐
│  Ingestion      │  │  Retrieval                      │
│  pipeline.py    │  │  ┌──────────────┐               │
│  ┌───────────┐  │  │  │ embed_query  │◄── Redis cache│
│  │  Parser   │  │  │  └──────┬───────┘               │
│  │(PDF/DOCX/ │  │  │         │ (parallel)             │
│  │  text)    │  │  │  ┌──────┴────────────────────┐  │
│  └───────────┘  │  │  │asyncio.gather             │  │
│  ┌───────────┐  │  │  │ dense_search  sparse_search│  │
│  │  Chunker  │  │  │  │ (Qdrant)      (Qdrant BM25)│  │
│  │ (NLTK +   │  │  │  └──────────────────────────┘  │
│  │ tiktoken) │  │  │         │                       │
│  └───────────┘  │  │  ┌──────▼───────┐               │
│  ┌───────────┐  │  │  │  RRF fusion  │               │
│  │  Embedder │  │  │  └──────┬───────┘               │
│  │  (Cohere  │  │  │         │                       │
│  │  embed-v3)│  │  │  ┌──────▼───────┐               │
│  └───────────┘  │  │  │ Cohere Rerank│               │
│  ┌───────────┐  │  │  │  v3 + CB     │               │
│  │  Sparse   │  │  │  └──────┬───────┘               │
│  │  Encoder  │  │  │         │                       │
│  │ (fastembed│  │  │  ┌──────▼───────┐               │
│  │   BM25)   │  │  │  │Context Packer│               │
│  └───────────┘  │  │  │(token budget)│               │
└─────────────────┘  │  └──────┬───────┘               │
         │           │         │                       │
         ▼           │  ┌──────▼───────┐               │
┌────────────────┐   │  │ Command R+   │               │
│  Qdrant        │◄──┘  │  documents=  │               │
│  - Dense vec   │      └──────────────┘               │
│  - Sparse vec  │            │                        │
│  - ACL payload │            ▼                        │
│    index       │       {answer, citations,           │
└────────────────┘        model_versions,              │
                          latency_ms}                  │
                                                       │
┌────────────────────────────────────────────────────┐ │
│  Observability                                     │ │
│  - structlog (JSON in prod)                        │ │
│  - Prometheus /metrics                             │ │
│  - OpenTelemetry → Jaeger                          │ │
│  - Audit log (query_hash, never raw query)         │ │
└────────────────────────────────────────────────────┘ │
```

## Architecture Decision Records

### ADR-001: Qdrant for hybrid search (single service)
**Status:** Accepted
**Decision:** Use Qdrant's native sparse+dense vector support instead of Elasticsearch.
**Rationale:** Qdrant 1.9 supports both dense and sparse vectors in a single collection, eliminating the need for an Elasticsearch sidecar. The Rust implementation has a significantly lower memory footprint. This aligns with the JD's "low-resource environments" requirement.
**Consequences:** fastembed must be used for BM25 vector generation (not BM25F). BM25 quality is close to Elasticsearch for most corpora.

### ADR-002: RRF over score-calibrated fusion
**Status:** Accepted
**Decision:** Use Reciprocal Rank Fusion (RRF) to merge dense and sparse results.
**Rationale:** Dense and sparse scores are not on the same scale. Score normalisation is fragile across different query types. RRF is rank-based, making it robust to score distribution differences. The k=60 constant from Cormack et al. (2009) is empirically validated across many domains.
**Consequences:** `alpha` parameter is kept in the API for documentation but does not affect RRF math. If score-based fusion is needed later, it can be added without breaking the API.

### ADR-003: ACL enforcement at Qdrant layer
**Status:** Accepted
**Decision:** ACL filtering is applied as a Qdrant payload filter, not in application code.
**Rationale:** Application-layer ACL has a window where retrieval completes before the ACL check. A bug in this window could leak documents. Qdrant payload filters are atomic with the vector search — no intermediate state where unfiltered results exist. The `acl_groups` field is indexed as a keyword payload for fast filtering.
**Consequences:** ACL groups must be stored in each Qdrant point's payload at ingest time. ACL changes to existing documents require re-indexing (acceptable for regulated industry use cases).

### ADR-004: Jinja2 templates for prompts
**Status:** Accepted
**Decision:** Use Jinja2 for all prompt construction, not f-strings.
**Rationale:** Jinja2 templates are stored in version control, diff-able, and testable in isolation. f-strings mix logic and template, making injection attacks harder to audit. StrictUndefined catches missing variables at render time rather than silently producing malformed prompts.
**Consequences:** Jinja2 is a runtime dependency (~180KB). Acceptable trade-off.

### ADR-005: Reranker circuit breaker
**Status:** Accepted
**Decision:** Wrap Cohere Rerank in a circuit breaker that falls back to RRF order on failure.
**Rationale:** A Cohere API outage should degrade answer quality, not take down query serving. The hybrid RRF score is a reasonable fallback that already outperforms BM25 alone.
**Consequences:** On fallback, `rerank_used: false` is returned in the response. Monitoring must alert on elevated `rag_rerank_fallback_total` to detect sustained Cohere outages.

### ADR-006: sha256(query) in audit log, not raw text
**Status:** Accepted
**Decision:** Audit events store `query_hash = sha256(raw_query)`, never `query = raw_query`.
**Rationale:** Raw queries are PII in regulated industries. A doctor querying "patient John Smith cancer diagnosis" — that's a medical record. Hashing allows correlation without exposing content. Security teams can verify a specific query hash if given the original query (pre-image verification).
**Consequences:** Audit logs cannot be used for query analysis or debugging without the original query. Operations teams must use separate (non-audit) debug logs gated by consent policies.
