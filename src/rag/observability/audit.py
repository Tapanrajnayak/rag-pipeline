"""Immutable audit log — hashes query text (PII), never logs raw queries.

Every query produces one audit event. Events are append-only by convention:
the logger never overwrites or deletes records.

In production: ship to a WORM (write-once, read-many) log store such as
AWS CloudTrail, Google Cloud Audit Logs, or an append-only Kafka topic.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

_audit_log = structlog.get_logger("audit")


@dataclass
class QueryAuditEvent:
    """Audit record for a single query.

    Critically: raw query text is NEVER stored.
    query_hash = sha256(raw_query) lets security teams correlate events
    without exposing PII (patient names, account numbers, etc.).
    """

    event_type: str = "query"
    user_id: str = ""
    query_hash: str = ""       # sha256(raw_query), NOT raw_query
    document_ids: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    rerank_used: bool = False
    chunk_count: int = 0
    timestamp_utc: float = field(default_factory=time.time)
    request_id: str = ""


def hash_query(query: str) -> str:
    """Return hex-encoded sha256 of the query string.

    Args:
        query: Raw query text (NEVER stored).

    Returns:
        64-character hex string.
    """
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def log_query_event(
    query: str,
    *,
    user_id: str,
    document_ids: list[str],
    model_versions: dict[str, str],
    latency_ms: float,
    rerank_used: bool,
    chunk_count: int,
    request_id: str,
) -> None:
    """Write an immutable audit event for a query.

    Args:
        query: Raw query text — hashed, never logged directly.
        user_id: Authenticated user identifier.
        document_ids: IDs of documents included in the response.
        model_versions: Dict mapping role to model name
                        (e.g. {'embed': 'embed-english-v3.0'}).
        latency_ms: Total query latency in milliseconds.
        rerank_used: Whether reranker was applied.
        chunk_count: Number of chunks in the final context.
        request_id: HTTP request correlation ID.
    """
    event = QueryAuditEvent(
        user_id=user_id,
        query_hash=hash_query(query),
        document_ids=document_ids,
        model_versions=model_versions,
        latency_ms=latency_ms,
        rerank_used=rerank_used,
        chunk_count=chunk_count,
        request_id=request_id,
    )

    _audit_log.info(
        "query_audit",
        event_type=event.event_type,
        user_id=event.user_id,
        query_hash=event.query_hash,        # ✓ hashed
        document_ids=event.document_ids,
        model_versions=event.model_versions,
        latency_ms=round(event.latency_ms, 2),
        rerank_used=event.rerank_used,
        chunk_count=event.chunk_count,
        timestamp_utc=event.timestamp_utc,
        request_id=event.request_id,
        # NEVER include: query=query  ← raw PII
    )


@dataclass
class IngestAuditEvent:
    """Audit record for a document ingest operation."""

    event_type: str = "ingest"
    user_id: str = ""
    document_id: str = ""
    filename_hash: str = ""    # sha256(filename) to avoid PII leakage
    acl_groups: list[str] = field(default_factory=list)
    chunks_stored: int = 0
    latency_ms: float = 0.0
    request_id: str = ""
    timestamp_utc: float = field(default_factory=time.time)


def log_ingest_event(
    filename: str,
    *,
    user_id: str,
    document_id: str,
    acl_groups: list[str],
    chunks_stored: int,
    latency_ms: float,
    request_id: str,
) -> None:
    """Write an immutable audit event for a document ingest.

    Args:
        filename: Original filename — hashed, never logged directly.
        user_id: Authenticated user identifier.
        document_id: Assigned document UUID.
        acl_groups: ACL groups applied to the document.
        chunks_stored: Number of chunks created.
        latency_ms: Total ingest latency in milliseconds.
        request_id: HTTP request correlation ID.
    """
    event = IngestAuditEvent(
        user_id=user_id,
        document_id=document_id,
        filename_hash=hashlib.sha256(filename.encode()).hexdigest(),
        acl_groups=acl_groups,
        chunks_stored=chunks_stored,
        latency_ms=latency_ms,
        request_id=request_id,
    )

    _audit_log.info(
        "ingest_audit",
        event_type=event.event_type,
        user_id=event.user_id,
        document_id=event.document_id,
        filename_hash=event.filename_hash,  # ✓ hashed
        acl_groups=event.acl_groups,
        chunks_stored=event.chunks_stored,
        latency_ms=round(event.latency_ms, 2),
        request_id=event.request_id,
        timestamp_utc=event.timestamp_utc,
    )
