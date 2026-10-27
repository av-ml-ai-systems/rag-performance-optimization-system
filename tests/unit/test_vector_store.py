# Module: test_vector_store
# Path: tests/test_vector_store.py
# Purpose: Validate the VectorStore component in isolation.
# Description: Test configuration validation, document storage, similarity
# search, and record counting using a mocked ChromaDB implementation.

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_performance.embedding_provider import EmbeddingProvider
from rag_performance.vector_store import VectorStore


@pytest.fixture
def embedding_provider() -> MagicMock:
    """Provide a mocked embedding provider."""
    return MagicMock(spec=EmbeddingProvider)


@pytest.fixture
def mock_chroma(request: pytest.FixtureRequest) -> MagicMock:
    """Provide a mocked ChromaDB class and register cleanup."""
    chroma_patcher = patch("rag_performance.vector_store.Chroma")
    chroma_mock = chroma_patcher.start()
    request.addfinalizer(chroma_patcher.stop)
    return chroma_mock


@pytest.fixture
def vector_store(
    embedding_provider: MagicMock,
    mock_chroma: MagicMock,
) -> VectorStore:
    """Create a VectorStore instance with mocked dependencies."""
    return VectorStore(
        embedding_provider=embedding_provider,
        collection_name="test_collection",
        persist_directory="test_chroma_db",
    )


def test_vector_store_initializes_with_configured_dependencies(
    embedding_provider: MagicMock,
    mock_chroma: MagicMock,
) -> None:
    """Verify configuration and injected dependencies are passed correctly."""
    VectorStore(
        embedding_provider=embedding_provider,
        collection_name="test_collection",
        persist_directory="test_chroma_db",
    )

    mock_chroma.assert_called_once_with(
        collection_name="test_collection",
        embedding_function=embedding_provider,
        persist_directory="test_chroma_db",
    )


@pytest.mark.parametrize(
    ("collection_name", "persist_directory", "error_message"),
    [
        ("", "test_chroma_db", "collection_name must not be empty"),
        ("   ", "test_chroma_db", "collection_name must not be empty"),
        ("test_collection", "", "persist_directory must not be empty"),
        ("test_collection", "   ", "persist_directory must not be empty"),
    ],
)
def test_vector_store_rejects_invalid_configuration(
    embedding_provider: MagicMock,
    mock_chroma: MagicMock,
    collection_name: str,
    persist_directory: str,
    error_message: str,
) -> None:
    """Verify invalid configuration is rejected."""
    with pytest.raises(ValueError, match=error_message):
        VectorStore(
            embedding_provider=embedding_provider,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    mock_chroma.assert_not_called()


def test_add_documents_stores_documents(
    vector_store: VectorStore,
    mock_chroma: MagicMock,
) -> None:
    """Verify documents are delegated to ChromaDB for storage."""
    documents = [
        Document(
            page_content="RAG retrieves relevant information.",
            metadata={"source": "rag_guide.md"},
        ),
        Document(
            page_content="Embeddings represent text as vectors.",
            metadata={"source": "embeddings.md"},
        ),
    ]

    vector_store.add_documents(documents)

    mock_chroma.return_value.add_documents.assert_called_once_with(documents=documents)


def test_add_documents_rejects_empty_list(
    vector_store: VectorStore,
    mock_chroma: MagicMock,
) -> None:
    """Verify an empty document list is rejected."""
    with pytest.raises(ValueError, match="documents must not be empty"):
        vector_store.add_documents([])

    mock_chroma.return_value.add_documents.assert_not_called()


def test_add_documents_rejects_blank_content(
    vector_store: VectorStore,
    mock_chroma: MagicMock,
) -> None:
    """Verify documents with blank content are rejected."""
    documents = [
        Document(page_content="Valid content."),
        Document(page_content="   "),
    ]

    with pytest.raises(ValueError, match="Document content must not be empty"):
        vector_store.add_documents(documents)

    mock_chroma.return_value.add_documents.assert_not_called()


def test_similarity_search_returns_matching_documents(
    vector_store: VectorStore,
    mock_chroma: MagicMock,
) -> None:
    """Verify similarity search returns ChromaDB results."""
    expected_documents = [
        Document(
            page_content="Relevant retrieval information.",
            metadata={"source": "rag_guide.md"},
        )
    ]
    mock_chroma.return_value.similarity_search.return_value = expected_documents

    results = vector_store.similarity_search(
        query="How does retrieval work?",
        k=3,
    )

    assert results == expected_documents
    mock_chroma.return_value.similarity_search.assert_called_once_with(
        query="How does retrieval work?",
        k=3,
    )


@pytest.mark.parametrize(
    ("query", "k", "error_message"),
    [
        ("", 4, "query must not be empty"),
        ("   ", 4, "query must not be empty"),
        ("valid query", 0, "k must be greater than or equal to 1"),
        ("valid query", -1, "k must be greater than or equal to 1"),
    ],
)
def test_similarity_search_rejects_invalid_arguments(
    vector_store: VectorStore,
    mock_chroma: MagicMock,
    query: str,
    k: int,
    error_message: str,
) -> None:
    """Verify invalid search arguments are rejected."""
    with pytest.raises(ValueError, match=error_message):
        vector_store.similarity_search(query=query, k=k)

    mock_chroma.return_value.similarity_search.assert_not_called()


def test_count_documents_returns_record_count(
    vector_store: VectorStore,
    mock_chroma: MagicMock,
) -> None:
    """Verify the number of stored records is returned."""
    mock_chroma.return_value.get.return_value = {
        "ids": ["record-1", "record-2", "record-3"]
    }

    result = vector_store.count_documents()

    assert result == 3
    mock_chroma.return_value.get.assert_called_once_with(include=[])
