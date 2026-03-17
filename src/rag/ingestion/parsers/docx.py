"""DOCX parser using python-docx."""

from __future__ import annotations

import io

import docx

from rag.core.errors import DocumentParseError


class DocxParser:
    """Extract text from DOCX bytes."""

    def parse(self, content: bytes, *, filename: str) -> str:
        """Extract text from a DOCX file.

        Args:
            content: Raw DOCX bytes.
            filename: Original filename (for error context).

        Returns:
            Paragraph text joined by newlines.

        Raises:
            DocumentParseError: if the document cannot be read.
        """
        try:
            doc = docx.Document(io.BytesIO(content))
        except Exception as exc:
            raise DocumentParseError(
                f"Failed to parse DOCX '{filename}'",
                detail=str(exc),
            ) from exc

        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise DocumentParseError(f"DOCX '{filename}' contains no text content.")

        return "\n\n".join(paragraphs)
