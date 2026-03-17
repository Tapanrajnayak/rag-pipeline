# Operations Runbook

## Quick Reference

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | Liveness — is the process up? |
| `GET /health/ready` | Readiness — are Qdrant + Redis healthy? |
| `GET /metrics` | Prometheus metrics |
| `http://localhost:16686` | Jaeger UI (traces) |
| `http://localhost:3000` | Grafana (metrics dashboard) |

---

## Scaling

### Horizontal scaling (stateless app tier)
The app is stateless — all persistent state is in Qdrant and Redis. To scale:
```bash
# Increase app replicas (docker-compose scale)
docker-compose up -d --scale app=3
```
Add a load balancer (nginx, Traefik) in front. The `X-Request-ID` header is forwarded through for request correlation.

### Qdrant scaling
Qdrant supports sharding for large collections. For this use case (<10M documents), a single node is sufficient. For larger corpora:
- Enable Qdrant's distributed mode
- Shard by document ACL group for optimal filtering performance

### Redis scaling
The embedding cache and rate limiter both tolerate Redis restarts (cache misses → Cohere API calls, rate limit window resets). For HA:
- Use Redis Sentinel or Redis Cluster
- TTLs self-expire stale data; no manual cleanup needed

---

## Common Errors

### `VectorStoreError: Failed to initialise Qdrant collection`
**Cause:** Qdrant unreachable at startup.
**Fix:**
```bash
curl http://localhost:6333/readyz
docker-compose restart qdrant
```

### `AuthenticationError: Token has expired`
**Cause:** JWT TTL exceeded (default 1hr).
**Fix:** Issue a new token. Check client token refresh logic.

### `EmbeddingError: Cohere embed API call failed`
**Cause:** Invalid API key, rate limit, or Cohere outage.
**Fix:** Check `COHERE_API_KEY`. Check Cohere status page. Embedding errors fail the ingest/query — they do not fall back silently.

### `rag_rerank_fallback_total` metric increasing
**Cause:** Cohere Rerank API returning 503 or timing out.
**Effect:** Queries continue to succeed (circuit breaker), but answers may have slightly lower relevance.
**Fix:** Check Cohere status. Alert if `rerank_fallback_total` rate > 5% of `query_total`.

### Rate limit 429s
**Cause:** Client exceeding 60 req/min (default).
**Fix:** Increase `API_RATE_LIMIT` env var, or add per-client exemptions.

---

## Circuit Breaker Behaviour

The reranker circuit breaker activates on any Cohere API error (503, timeout, connection refused). When active:
- Queries succeed with hybrid RRF ranking instead of cross-encoder ranking
- `rerank_used: false` in every response
- `rag_rerank_fallback_total` counter increments

Monitor with:
```promql
rate(rag_rerank_fallback_total[5m]) / rate(rag_query_total[5m])
```
Alert when fallback rate > 0.05 (5%).

---

## Monitoring Alerts

| Alert | PromQL | Threshold |
|---|---|---|
| High query latency | `histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m]))` | > 2s |
| Rerank fallback | `rate(rag_rerank_fallback_total[5m]) / rate(rag_query_total[5m])` | > 0.05 |
| High error rate | `rate(rag_query_total{status="error"}[5m]) / rate(rag_query_total[5m])` | > 0.01 |
| Low cache hit ratio | `rate(rag_embedding_cache_hits_total[5m]) / (rate(rag_embedding_cache_hits_total[5m]) + rate(rag_embedding_cache_misses_total[5m]))` | < 0.5 |

---

## GDPR Erasure Procedure

To delete a document (right-to-erasure request):
```bash
# Requires admin JWT
curl -X DELETE \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/v1/documents/{document_id}
```

This deletes:
- All Qdrant vector points for the document
- Document metadata from the document store

The audit log entry for the deletion is retained (immutable, does not contain PII).

---

## Key Metrics

```bash
# Query duration p95
curl -s http://localhost:8000/metrics | grep rag_query_duration

# Cache hit ratio
curl -s http://localhost:8000/metrics | grep rag_embedding_cache

# Rerank fallback count
curl -s http://localhost:8000/metrics | grep rag_rerank_fallback
```
