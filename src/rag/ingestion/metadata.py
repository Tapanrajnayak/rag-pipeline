"""Document-level metadata and ACL tag structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DocumentMetadata:
    """Metadata attached to every document at ingest time."""

    document_id: str
    filename: str
    content_type: str           # "application/pdf", "text/plain", etc.
    acl_groups: list[str]       # groups that may access this document
    uploaded_by: str            # user_id of uploader
    uploaded_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    title: str | None = None
    source_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize to Qdrant point payload format."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "acl_groups": self.acl_groups,
            "uploaded_by": self.uploaded_by,
            "uploaded_at": self.uploaded_at.isoformat(),
            "title": self.title,
            "source_url": self.source_url,
            **self.extra,
        }


@dataclass
class ChunkMetadata:
    """Metadata for an individual chunk within a document."""

    document_id: str
    chunk_index: int
    start_sentence: int
    end_sentence: int
    token_count: int
    acl_groups: list[str]       # inherited from parent document

    def to_qdrant_payload(self, doc_meta: DocumentMetadata) -> dict[str, Any]:
        """Merge chunk metadata with document metadata for the Qdrant payload."""
        payload = doc_meta.to_qdrant_payload()
        payload.update(
            {
                "chunk_index": self.chunk_index,
                "start_sentence": self.start_sentence,
                "end_sentence": self.end_sentence,
                "token_count": self.token_count,
            }
        )
        return payload
