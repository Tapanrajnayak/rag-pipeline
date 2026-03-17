# Security Model

## Threat Model

### Assets
- Document contents (may be proprietary, legally privileged, or contain PII)
- User query text (may contain PII — patient names, account numbers)
- API keys and JWT signing secrets

### Threats and Controls

| Threat | Control |
|---|---|
| Unauthorised document access | JWT auth + Qdrant ACL payload filter (atomic with retrieval) |
| JWT forgery | HMAC-SHA256 with minimum 32-byte secret; audience + issuer validation |
| Expired token reuse | `exp` claim validated; clock drift tolerance = 0 |
| PII leakage via audit logs | Query text hashed (sha256); audit log contains only `query_hash` |
| Secret exposure via logs | All secrets are `pydantic.SecretStr`; repr and JSON are redacted |
| Prompt injection | Jinja2 templates with `autoescape`-safe rendering; user input in `message=` only |
| Container privilege escalation | Non-root user (uid 1000); no shell in final image |
| Dependency vulnerabilities | Weekly `uv audit` + `trivy image` in CI |
| Secret leakage in code | `detect-secrets` pre-commit hook + CI scan |

---

## ACL Design

ACL groups are stored in every Qdrant vector point's payload:
```json
{
  "acl_groups": ["engineering", "all"]
}
```

The `acl_groups` field is indexed as a Qdrant keyword payload index for efficient filtering.

Every query applies a Qdrant filter before any vector scoring:
```python
{
  "must": [
    {
      "key": "acl_groups",
      "match": {"any": user.acl_groups}  # e.g. ["engineering", "all"]
    }
  ]
}
```

**Critical:** This filter is atomic with retrieval. There is no application-side window where documents can be retrieved before ACL is applied.

The `all` group is a sentinel added to every user's group list and every document's ACL at creation. A document tagged `["all"]` is accessible to every authenticated user.

---

## Data Flow

```
User query
    │
    ▼
JWT validation (audience + issuer + expiry)
    │ success: UserContext{user_id, groups}
    ▼
embed_query → Redis cache (key: sha256(query + model + input_type))
    │
    ▼
Qdrant search WITH ACL filter (atomic — no intermediate unfiltered state)
    │
    ▼
Rerank (Cohere API — query + chunk text only, no user identity)
    │
    ▼
Context packer (token budget enforcement)
    │
    ▼
Cohere Command R+ generate (query + documents only)
    │
    ▼
Audit log: {query_hash, user_id, doc_ids, model_versions, latency_ms}
           NOT: {query, doc_contents}
    │
    ▼
Response: {answer, citations, model_versions, latency_ms}
```

---

## What We Never Log

| Data | Why |
|---|---|
| Raw query text | May contain PII (patient names, account numbers) |
| Authorization header | Contains JWT token |
| Request/response bodies | May contain document content or PII |
| Document text in non-audit logs | Proprietary content |

We DO log:
- `query_hash = sha256(raw_query)` in audit log
- `user_id` (not email) in all logs
- `document_id` (not document text) in audit log
- HTTP method, path, status code, latency in access log
