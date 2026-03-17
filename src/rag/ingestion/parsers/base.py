"""Parser Protocol — structural subtyping; no inheritance required."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Parser(Protocol):
    """Any object with this method signature is a valid parser.

    Structural subtyping means downstream code (pipeline, tests) depends on
    this Protocol, not on any specific implementation. Callers can inject any
    compatible object without importing this module.
    """

    def parse(self, content: bytes, *, filename: str) -> str:
        """Extract plain text from raw file bytes.

        Args:
            content: Raw file bytes.
            filename: Original filename (used for MIME-type detection fallback).

        Returns:
            Extracted plain text. Whitespace may be irregular; the chunker
            normalises it.

        Raises:
            DocumentParseError: if the content cannot be parsed.
        """
        ...
