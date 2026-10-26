# Module: retriever
# Path: src/rag_performance/retriever.py
# Purpose: Retrieve relevant document chunks for user queries.
# Description: Coordinate document retrieval through an injected VectorStore
# while keeping retrieval configuration and input validation centralized.

from langchain_core.documents import Document

from rag_performance.vector_store import VectorStore


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

    def retrieve(self, query: str) -> list[Document]:
        """Return the most relevant documents for a non-empty query."""
        if not query.strip():
            raise ValueError("query must not be empty.")

        return self._vector_store.similarity_search(
            query=query,
            k=self._k,
        )
