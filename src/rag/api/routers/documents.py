"""Documents API — ingest, list, delete."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from rag.api.dependencies import get_current_user, get_doc_repo, get_embedder, get_sparse_encoder, get_vector_store
from rag.api.schemas.documents import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestResponse,
)
from rag.core.config import Settings, get_settings
from rag.core.security import UserContext
from rag.ingestion.pipeline import ingest_document
from rag.ingestion.parsers.docx import DocxParser
from rag.ingestion.parsers.pdf import PdfParser
from rag.ingestion.parsers.text import TextParser
from rag.observability.audit import log_ingest_event
from rag.observability.metrics import INGEST_DURATION, INGEST_TOTAL

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_PARSERS = {
    "application/pdf": PdfParser(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxParser(),
    "text/plain": TextParser(),
}


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    acl_groups: str = Form(default="all", description="Comma-separated ACL group names"),
    user: UserContext = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    vector_store=Depends(get_vector_store),  # type: ignore[no-untyped-def]
    doc_repo=Depends(get_doc_repo),  # type: ignore[no-untyped-def]
    embedder=Depends(get_embedder),  # type: ignore[no-untyped-def]
    sparse_encoder=Depends(get_sparse_encoder),  # type: ignore[no-untyped-def]
) -> IngestResponse:
    """Ingest a document into the RAG pipeline.

    - Supported types: PDF, DOCX, plain text.
    - ACL groups control who can retrieve this document.
    - The 'all' group is always included implicitly.
    """
    content_type = file.content_type or "text/plain"
    parser = _PARSERS.get(content_type)
    if parser is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {content_type}. Supported: {list(_PARSERS)}",
        )

    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    if "all" not in groups:
        groups.append("all")

    content = await file.read()
    start = time.monotonic()

    try:
        result = await ingest_document(
            content,
            filename=file.filename or "upload",
            content_type=content_type,
            acl_groups=groups,
            uploaded_by=user.user_id,
            parser=parser,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            vector_store=vector_store,
            doc_repo=doc_repo,
        )
        latency_ms = (time.monotonic() - start) * 1000
        INGEST_TOTAL.labels(status="success").inc()
        INGEST_DURATION.observe(latency_ms / 1000)
    except Exception as exc:
        INGEST_TOTAL.labels(status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    request_id = getattr(request.state, "request_id", "")
    log_ingest_event(
        file.filename or "upload",
        user_id=user.user_id,
        document_id=result.document_id,
        acl_groups=groups,
        chunks_stored=result.chunks_stored,
        latency_ms=latency_ms,
        request_id=request_id,
    )

    return IngestResponse(
        document_id=result.document_id,
        chunks_stored=result.chunks_stored,
        filename=result.filename,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(
    request: Request,
    user: UserContext = Depends(get_current_user),
    doc_repo=Depends(get_doc_repo),  # type: ignore[no-untyped-def]
) -> DocumentListResponse:
    """List all documents the current user has access to."""
    all_docs = await doc_repo.list_all()
    # Filter to docs where user's groups overlap the doc's acl_groups
    accessible = [
        d for d in all_docs
        if any(g in user.acl_groups for g in d.acl_groups)
    ]

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                document_id=d.document_id,
                filename=d.filename,
                content_type=d.content_type,
                acl_groups=d.acl_groups,
                uploaded_by=d.uploaded_by,
                uploaded_at=d.uploaded_at,
                title=d.title,
            )
            for d in accessible
        ],
        total=len(accessible),
        request_id=getattr(request.state, "request_id", ""),
    )


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document (GDPR right-to-erasure)",
)
async def delete_document(
    document_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
    vector_store=Depends(get_vector_store),  # type: ignore[no-untyped-def]
    doc_repo=Depends(get_doc_repo),  # type: ignore[no-untyped-def]
) -> DeleteResponse:
    """Delete all chunks and metadata for a document.

    Only the uploader or an admin (group 'admin') can delete a document.
    """
    doc_meta = await doc_repo.get(document_id)
    if doc_meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )

    # Authorization check: must be the uploader or in admin group
    if doc_meta.uploaded_by != user.user_id and "admin" not in user.groups:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this document",
        )

    await vector_store.delete_document(document_id)
    deleted = await doc_repo.delete(document_id)

    return DeleteResponse(
        document_id=document_id,
        deleted=deleted,
        request_id=getattr(request.state, "request_id", ""),
    )
