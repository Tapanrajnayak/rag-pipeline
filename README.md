# RAG Pipeline

Production-grade Retrieval-Augmented Generation pipeline targeting regulated-industry enterprise use cases. Built end-to-end on the Cohere stack.

## What This Demonstrates

| Requirement | Implementation |
|---|---|
| Shipped Python in production | FastAPI + async throughout; mypy strict; ruff; pytest ≥85% coverage |
| Hybrid RAG | Qdrant dense + sparse (BM25) in one collection; async parallel retrieval; RRF fusion |
| Security & privacy | JWT auth; Qdrant ACL payload filter (atomic); audit log hashes PII; SecretStr; non-root container |
| Performant, low-resource | Rust vector engine; Redis embedding cache; multi-stage Docker image; explicit mem + CPU limits |
| Cohere end-to-end | embed-v3 (input_type discipline) → Rerank v3 (circuit breaker) → Command R+ (documents=) |

## Architecture

```
POST /v1/query
    │
    ├─→ JWT validation → UserContext{user_id, groups}
    ├─→ embed_query (Redis-cached, sha256 key)
    ├─→ BM25 sparse encode
    ├─→ asyncio.gather(dense_search, sparse_search) ← ACL filter atomic with search
    ├─→ RRF fusion: score(d) = Σ 1/(k=60 + rank_i)
    ├─→ Cohere Rerank v3 (circuit breaker → RRF fallback on 503)
    ├─→ Token-budget context packer (sentence-boundary truncation)
    ├─→ Cohere Command R+ (documents=, temperature=0)
    └─→ {answer, citations, model_versions, latency_ms, request_id}
```

See [docs/architecture.md](docs/architecture.md) for ADRs and full system diagram.

## Stack

| Component | Choice |
|---|---|
| Embeddings | Cohere embed-english-v3.0 (`input_type` discipline) |
| Vector store | Qdrant (native sparse+dense, Rust, low memory) |
| Sparse retrieval | fastembed BM25 (no Elasticsearch) |
| Reranking | Cohere Rerank v3 + circuit breaker |
| Generation | Cohere Command R+ (`documents=` for grounded answers) |
| Cache | Redis (query embedding cache, rate limiter) |
| API | FastAPI + uvicorn |
| Observability | structlog + Prometheus + OpenTelemetry → Jaeger |

## Quick Start

```bash
# 1. Copy and fill environment
cp .env.example .env
# Set COHERE_API_KEY and JWT_SECRET_KEY

# 2. Start infrastructure
docker-compose -f infra/docker-compose.yml up -d

# 3. Check readiness
curl http://localhost:8000/health/ready
# → {"status":"ready","qdrant":"ok","redis":"ok"}

# 4. Seed demo corpus (3 ACL groups: engineering / legal / all)
uv run python scripts/seed_demo.py

# 5. Generate a test JWT
python3 -c "
import os
from rag.core.security import create_access_token
token = create_access_token(
    'demo-user', ['engineering', 'all'],
    secret_key=os.environ['JWT_SECRET_KEY'],
    algorithm='HS256',
    audience='rag-pipeline',
    issuer='rag-pipeline-auth',
)
print(token)
"

# 6. Query
curl -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "tradeoffs of hybrid vs pure semantic search"}' \
     http://localhost:8000/v1/query

# 7. Benchmark
uv run python scripts/benchmark.py --queries 100 --concurrency 10
```

## Development

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run unit tests (no I/O required)
uv run pytest tests/unit/ -v

# Run all tests (requires Qdrant + Redis)
docker-compose -f infra/docker-compose.test.yml up -d
uv run pytest -v

# Lint + type-check
uv run ruff check src/ tests/
uv run mypy src/rag/
```

## Security

- JWT validation: audience, issuer, expiry — all enforced
- ACL at Qdrant layer: filtering is atomic with retrieval (no app-layer bypass window)
- Audit log: `query_hash = sha256(query)` — never raw query text
- `pydantic.SecretStr` for all secrets — redacted in repr and JSON
- Non-root container (uid 1000), no shell in final image
- Weekly `uv audit` + `trivy` + `bandit` + `semgrep` in CI

See [docs/security.md](docs/security.md) for full threat model.

## Observability

| Signal | Endpoint |
|---|---|
| Prometheus metrics | `http://localhost:8000/metrics` |
| Grafana dashboard | `http://localhost:3000` |
| Jaeger traces | `http://localhost:16686` |

Key metrics:
- `rag_query_duration_seconds` — p50/p95/p99 query latency
- `rag_rerank_fallback_total` — circuit breaker activations
- `rag_embedding_cache_hit_ratio` — Redis cache effectiveness
- `rag_document_count` — per ACL group

## Acceptance Criteria

- [x] `mypy --strict` zero errors
- [x] `pytest` ≥85% coverage, zero skips in unit + integration
- [x] ACL security test suite passes completely
- [x] Audit log: query_hash only, never raw query text
- [x] Circuit breaker: reranker failure degrades quality, never kills queries
- [x] Docker image target: <300MB (multi-stage, non-root, no dev tools)
- [x] p95 query latency target: ≤2s on local hardware

## License

MIT
