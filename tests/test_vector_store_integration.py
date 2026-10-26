# Module: test_vector_store_integration
# Path: tests/test_vector_store_integration.py
# Purpose: Validate VectorStore integration with a real ChromaDB collection.
# Description: Verify document storage, metadata preservation, similarity
# search, record counting, and persistence using deterministic test embeddings.

from pathlib import Path

from langchain_core.documents import Document

from rag_performance.embedding_provider import EmbeddingProvider
from rag_performance.vector_store import VectorStore


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Generate deterministic vectors for isolated integration tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return a vector for each document text."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a vector for a search query."""
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Map text into a simple deterministic vector."""
        normalized_text = text.lower()

        return [
            float("retrieval" in normalized_text or "search" in normalized_text),
            float("benefits" in normalized_text or "insurance" in normalized_text),
            float("travel" in normalized_text or "expenses" in normalized_text),
        ]


def test_vector_store_persists_and_retrieves_documents(
    tmp_path: Path,
) -> None:
    """Verify storage, metadata, search, counting, and persistence."""
    persist_directory = str(tmp_path / "chroma_db")
    embedding_provider = DeterministicEmbeddingProvider()

    documents = [
        Document(
            page_content="Retrieval search finds relevant knowledge.",
            metadata={"source": "retrieval.md", "topic": "retrieval"},
        ),
        Document(
            page_content="Employee benefits include health insurance.",
            metadata={"source": "benefits.md", "topic": "benefits"},
        ),
        Document(
            page_content="Business travel expenses require approval.",
            metadata={"source": "travel.md", "topic": "travel"},
        ),
    ]

    vector_store = VectorStore(
        embedding_provider=embedding_provider,
        collection_name="integration_test_collection",
        persist_directory=persist_directory,
    )

    vector_store.add_documents(documents)

    assert vector_store.count_documents() == 3

    results = vector_store.similarity_search(
        query="How does retrieval search work?",
        k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "retrieval.md"
    assert results[0].metadata["topic"] == "retrieval"

    reopened_vector_store = VectorStore(
        embedding_provider=embedding_provider,
        collection_name="integration_test_collection",
        persist_directory=persist_directory,
    )

    assert reopened_vector_store.count_documents() == 3

    persisted_results = reopened_vector_store.similarity_search(
        query="How does retrieval search work?",
        k=1,
    )

    assert len(persisted_results) == 1
    assert persisted_results[0].metadata["source"] == "retrieval.md"
