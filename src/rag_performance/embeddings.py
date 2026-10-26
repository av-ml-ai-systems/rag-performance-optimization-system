# Module: embeddings
# Path: src/rag_performance/embeddings.py
# Purpose: Generate embeddings through a locally hosted Ollama model.
# Description: Implement the shared EmbeddingProvider contract while keeping
# Ollama-specific behavior isolated from the rest of the RAG architecture.

from langchain_ollama import OllamaEmbeddings

from rag_performance.embedding_provider import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate document and query embeddings using Ollama."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
    ) -> None:
        """Initialize the provider with the selected Ollama model."""
        if not model.strip():
            raise ValueError("model must not be empty.")

        self._embeddings = OllamaEmbeddings(model=model)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple document texts."""
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts must not be empty.")

        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a search query."""
        if not text.strip():
            raise ValueError("Query text must not be empty.")

        return self._embeddings.embed_query(text)
