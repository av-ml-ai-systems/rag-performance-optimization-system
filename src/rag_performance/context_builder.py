# Module: context_builder
# Path: src/rag_performance/context_builder.py
# Purpose: Build structured context from retrieved document chunks.
# Description: Combine document content with available source metadata
# into a consistent context string for downstream prompt construction.

from langchain_core.documents import Document


class ContextBuilder:
    """Build context from retrieved documents."""

    def build(self, documents: list[Document]) -> str:
        """Format documents into a context string with source attribution."""
        if not documents:
            return ""

        context_sections: list[str] = []

        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "Unknown source")
            content = document.page_content.strip()

            if not content:
                continue

            context_sections.append(f"[Document {index} | Source: {source}]\n{content}")

        return "\n\n".join(context_sections)
