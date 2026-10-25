# Module: chunking
# Path: src/rag_performance/chunking.py
# Purpose: Split ingested documents into smaller chunks for RAG retrieval.
# Description: Use LangChain's recursive character splitter while preserving
# document metadata and validating chunking configuration and inputs.

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Split documents into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the chunker with validated configuration."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Split documents into chunks while preserving their metadata."""
        if not documents:
            return []

        for document in documents:
            if not document.page_content.strip():
                raise ValueError("Input documents must not contain empty text.")

        return self._splitter.split_documents(documents)
