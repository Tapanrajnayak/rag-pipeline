"""Plain text parser — UTF-8 with fallback to latin-1."""

from __future__ import annotations

from rag.core.errors import DocumentParseError


class TextParser:
    """Parse plain text bytes into a string."""

    def parse(self, content: bytes, *, filename: str) -> str:
        """Decode raw bytes as UTF-8 text with latin-1 fallback.

        Args:
            content: Raw text bytes.
            filename: Original filename (for error context).

        Returns:
            Decoded string content.

        Raises:
            DocumentParseError: if the content cannot be decoded.
        """
        for encoding in ("utf-8", "latin-1"):
            try:
                text = content.decode(encoding).strip()
                if text:
                    return text
            except (UnicodeDecodeError, ValueError):
                continue

        raise DocumentParseError(
            f"Could not decode text file '{filename}' with UTF-8 or latin-1 encoding."
        )
