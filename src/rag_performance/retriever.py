# Module: retriever
# Path: src/rag_performance/retriever.py
# Purpose: Retrieve relevant document chunks for user queries.
# Description: Coordinate document retrieval through an injected VectorStore
# while keeping retrieval configuration and input validation centralized.

from __future__ import annotations

from langchain_core.documents import Document

from rag_performance.logging import get_logger
from rag_performance.vector_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """Retrieve relevant documents from a vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 4,
    ) -> None:
        """Initialize the retriever with a vector store and result limit."""
        if k < 1:
            raise ValueError("k must be greater than or equal to 1.")

        self._vector_store = vector_store
        self._k = k

    @property
    def k(self) -> int:
        """Return the configured retrieval limit k."""
        return self._k

    def retrieve(self, query: str) -> list[Document]:
        """Return the most relevant documents for a non-empty query."""
        if not query.strip():
            logger.warning("Retrieval rejected: query must not be empty")
            raise ValueError("query must not be empty.")

        logger.debug("Executing similarity search (k=%d)", self._k)

        documents = self._vector_store.similarity_search(
            query=query,
            k=self._k,
        )

        logger.info(
            "Vector search completed (documents_retrieved=%d, k=%d)",
            len(documents),
            self._k,
        )

        return documents
