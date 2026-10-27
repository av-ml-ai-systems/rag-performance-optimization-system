# Module: test_embeddings_integration
# Path: tests/test_embeddings_integration.py
# Purpose: Verify real embedding generation through the local Ollama service.
# Description: Exercise document and query embedding with the configured model
# to confirm that the provider integrates correctly with Ollama.

from rag_performance.embeddings import OllamaEmbeddingProvider


def test_ollama_generates_document_embeddings() -> None:
    """Verify Ollama generates valid vectors for document texts."""
    provider = OllamaEmbeddingProvider()

    vectors = provider.embed_documents(
        [
            "RAG retrieves relevant documents to support an answer.",
            "Embeddings represent text as numerical vectors.",
        ]
    )

    assert len(vectors) == 2
    assert all(vector for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)
    assert len(vectors[0]) == len(vectors[1])


def test_ollama_generates_query_embedding() -> None:
    """Verify Ollama generates a valid vector for a search query."""
    provider = OllamaEmbeddingProvider()

    vector = provider.embed_query("How does retrieval-augmented generation work?")

    assert vector
    assert all(isinstance(value, float) for value in vector)
