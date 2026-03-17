"""Request/response schemas for the documents API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Response schema for a single document."""

    document_id: str
    filename: str
    content_type: str
    acl_groups: list[str]
    uploaded_by: str
    uploaded_at: datetime
    title: str | None = None


class IngestResponse(BaseModel):
    """Response after successfully ingesting a document."""

    document_id: str
    chunks_stored: int
    filename: str
    request_id: str
    latency_ms: float


class DeleteResponse(BaseModel):
    """Response after deleting a document."""

    document_id: str
    deleted: bool
    request_id: str


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    documents: list[DocumentResponse]
    total: int
    request_id: str
