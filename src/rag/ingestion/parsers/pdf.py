"""PDF parser using pypdf."""

from __future__ import annotations

import io

from pypdf import PdfReader

from rag.core.errors import DocumentParseError


class PdfParser:
    """Extract text from PDF bytes."""

    def parse(self, content: bytes, *, filename: str) -> str:
        """Extract text from a PDF file.

        Args:
            content: Raw PDF bytes.
            filename: Original filename (for error context).

        Returns:
            Concatenated plain text from all pages.

        Raises:
            DocumentParseError: if the PDF cannot be read or has no text.
        """
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise DocumentParseError(
                f"Failed to parse PDF '{filename}'",
                detail=str(exc),
            ) from exc

        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        if not pages:
            raise DocumentParseError(
                f"PDF '{filename}' contains no extractable text. "
                "It may be a scanned document requiring OCR."
            )

        return "\n\n".join(pages)
