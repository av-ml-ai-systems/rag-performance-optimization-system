# Module: test_embeddings
# Path: tests/test_embeddings.py
# Purpose: Validate the embedding provider contract and Ollama implementation.
# Description: Test input validation, abstract-class behavior, and embedding
# generation using a mocked Ollama client without requiring model inference.

from unittest.mock import patch

import pytest

from rag_performance.embedding_provider import EmbeddingProvider
from rag_performance.embeddings import OllamaEmbeddingProvider


def test_embedding_provider_is_abstract() -> None:
    """Verify the embedding contract cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        type.__call__(EmbeddingProvider)


def test_ollama_embedding_provider_rejects_empty_model() -> None:
    """Verify an empty model name is rejected."""
    with pytest.raises(ValueError, match="model must not be empty"):
        OllamaEmbeddingProvider(model="  ")


def test_embed_documents_returns_empty_list_for_empty_input() -> None:
    """Verify empty document input does not invoke the Ollama client."""
    with patch("rag_performance.embeddings.OllamaEmbeddings") as mock_client:
        provider = OllamaEmbeddingProvider()
        result = provider.embed_documents([])

    assert result == []
    mock_client.return_value.embed_documents.assert_not_called()


def test_embed_documents_rejects_blank_text() -> None:
    """Verify blank document text is rejected before inference."""
    with patch("rag_performance.embeddings.OllamaEmbeddings"):
        provider = OllamaEmbeddingProvider()

    with pytest.raises(ValueError, match="Document texts must not be empty"):
        provider.embed_documents(["valid text", "   "])


def test_embed_documents_delegates_to_ollama() -> None:
    """Verify document texts are passed to Ollama and vectors returned."""
    expected_vectors = [[0.1, 0.2], [0.3, 0.4]]

    with patch("rag_performance.embeddings.OllamaEmbeddings") as mock_client:
        mock_client.return_value.embed_documents.return_value = expected_vectors
        provider = OllamaEmbeddingProvider()

        result = provider.embed_documents(["first document", "second document"])

    assert result == expected_vectors
    mock_client.return_value.embed_documents.assert_called_once_with(
        ["first document", "second document"]
    )


def test_embed_query_rejects_blank_text() -> None:
    """Verify blank query text is rejected before inference."""
    with patch("rag_performance.embeddings.OllamaEmbeddings"):
        provider = OllamaEmbeddingProvider()

    with pytest.raises(ValueError, match="Query text must not be empty"):
        provider.embed_query("  ")


def test_embed_query_delegates_to_ollama() -> None:
    """Verify query text is passed to Ollama and its vector is returned."""
    expected_vector = [0.1, 0.2, 0.3]

    with patch("rag_performance.embeddings.OllamaEmbeddings") as mock_client:
        mock_client.return_value.embed_query.return_value = expected_vector
        provider = OllamaEmbeddingProvider()

        result = provider.embed_query("How does retrieval work?")

    assert result == expected_vector
    mock_client.return_value.embed_query.assert_called_once_with(
        "How does retrieval work?"
    )


def test_embedding_provider_accepts_custom_model() -> None:
    """Verify a configured model name is passed to the Ollama client."""
    with patch("rag_performance.embeddings.OllamaEmbeddings") as mock_client:
        OllamaEmbeddingProvider(model="custom-embedding-model")

    mock_client.assert_called_once_with(model="custom-embedding-model")


def test_ollama_provider_implements_embedding_contract() -> None:
    """Verify the concrete provider implements the abstract contract."""
    with patch("rag_performance.embeddings.OllamaEmbeddings"):
        provider = OllamaEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    assert callable(provider.embed_documents)
    assert callable(provider.embed_query)
