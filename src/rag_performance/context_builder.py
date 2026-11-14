# Module: context_builder
# Path: src/rag_performance/context_builder.py
# Purpose: Build structured context from retrieved document chunks.
# Description: Combine document content with available source metadata
# into a consistent context string for downstream prompt construction.

from __future__ import annotations

from langchain_core.documents import Document

from rag_performance.logging import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    """Build context from retrieved documents."""

    def __init__(self) -> None:
        """Initialize the context builder with default metrics state."""
        self._last_chunks_used: int = 0

    @property
    def last_chunks_used(self) -> int:
        """Return number of chunks used in the most recent build."""
        return self._last_chunks_used

    def build(self, documents: list[Document]) -> str:
        """Format documents into a context string with source attribution."""
        if not documents:
            self._last_chunks_used = 0
            logger.warning("No documents provided for context construction")
            return ""

        context_sections: list[str] = []

        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "Unknown source")
            content = document.page_content.strip()

            if not content:
                logger.debug("Skipped empty chunk at index %d", index)
                continue

            context_sections.append(f"[Document {index} | Source: {source}]\n{content}")

        self._last_chunks_used = len(context_sections)
        context_result = "\n\n".join(context_sections)

        logger.info(
            "Context construction completed (chunks_used=%d, context_size_chars=%d)",
            self._last_chunks_used,
            len(context_result),
        )

        return context_result
