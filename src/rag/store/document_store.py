"""Lightweight in-memory document metadata store.

In production this would be backed by Postgres (asyncpg + SQLAlchemy async).
For the demo, an in-memory dict with the same interface keeps the dependency
surface minimal and tests blazing fast.

The interface is defined via a Protocol so callers are decoupled from the
implementation — swap to Postgres without touching any retrieval or API code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from rag.ingestion.metadata import DocumentMetadata


@runtime_checkable
class DocumentRepository(Protocol):
    """Persistent store for document metadata."""

    async def save(self, meta: DocumentMetadata) -> None:
        """Persist document metadata.

        Args:
            meta: Document metadata to save.
        """
        ...

    async def get(self, document_id: str) -> DocumentMetadata | None:
        """Retrieve metadata by document ID.

        Args:
            document_id: UUID string.

        Returns:
            DocumentMetadata if found, None otherwise.
        """
        ...

    async def list_all(self) -> list[DocumentMetadata]:
        """Return all stored document metadata records."""
        ...

    async def delete(self, document_id: str) -> bool:
        """Delete metadata for a document.

        Args:
            document_id: UUID string.

        Returns:
            True if deleted, False if not found.
        """
        ...


class InMemoryDocumentStore:
    """In-memory implementation of DocumentRepository for dev and testing."""

    def __init__(self) -> None:
        self._store: dict[str, DocumentMetadata] = {}

    async def save(self, meta: DocumentMetadata) -> None:
        self._store[meta.document_id] = meta

    async def get(self, document_id: str) -> DocumentMetadata | None:
        return self._store.get(document_id)

    async def list_all(self) -> list[DocumentMetadata]:
        return list(self._store.values())

    async def delete(self, document_id: str) -> bool:
        if document_id in self._store:
            del self._store[document_id]
            return True
        return False
