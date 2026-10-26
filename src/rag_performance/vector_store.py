# Module: vector_store
# Path: src/rag_performance/vector_store.py
# Purpose: Store and retrieve document chunks using ChromaDB.
# Description: Inject an embedding provider, persist vectors locally,
# and expose focused operations for document storage and similarity search.

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_performance.embedding_provider import EmbeddingProvider


class VectorStore:
    """Manage document storage and similarity search with ChromaDB."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "rag_knowledge_base",
        persist_directory: str = "data/chroma_db",
    ) -> None:
        """Initialize a persistent ChromaDB collection."""
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        if not persist_directory.strip():
            raise ValueError("persist_directory must not be empty.")

        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)

        self._vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_provider,
            persist_directory=str(self._persist_directory),
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and store document chunks with their metadata."""
        if not documents:
            raise ValueError("documents must not be empty.")

        if any(not document.page_content.strip() for document in documents):
            raise ValueError("Document content must not be empty.")

        self._vector_store.add_documents(documents=documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Return the most relevant documents for a query."""
        if not query.strip():
            raise ValueError("query must not be empty.")

        if k < 1:
            raise ValueError("k must be greater than or equal to 1.")

        return self._vector_store.similarity_search(query=query, k=k)

    def count_documents(self) -> int:
        """Return the number of records in the collection."""
        records = self._vector_store.get(include=[])

        return len(records["ids"])
