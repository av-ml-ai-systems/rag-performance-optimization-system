# Module: test_retriever
# Path: tests/test_retriever.py
# Purpose: Validate Retriever behavior independently of ChromaDB.
# Description: Test configuration, query validation, document retrieval,
# and delegation to the injected VectorStore.

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_performance.retriever import Retriever
from rag_performance.vector_store import VectorStore


@pytest.fixture
def vector_store() -> MagicMock:
    """Provide a mocked vector store."""
    return MagicMock(spec=VectorStore)


def test_retriever_uses_default_result_limit(
    vector_store: MagicMock,
) -> None:
    """Verify the retriever can be initialized with its default k."""
    retriever = Retriever(vector_store=vector_store)

    retriever.retrieve("How does retrieval work?")

    vector_store.similarity_search.assert_called_once_with(
        query="How does retrieval work?",
        k=4,
    )


def test_retriever_uses_configured_result_limit(
    vector_store: MagicMock,
) -> None:
    """Verify the configured k value is passed to the vector store."""
    retriever = Retriever(vector_store=vector_store, k=2)

    retriever.retrieve("How does retrieval work?")

    vector_store.similarity_search.assert_called_once_with(
        query="How does retrieval work?",
        k=2,
    )


@pytest.mark.parametrize("k", [0, -1])
def test_retriever_rejects_invalid_result_limit(
    vector_store: MagicMock,
    k: int,
) -> None:
    """Verify k must be a positive integer."""
    with pytest.raises(
        ValueError,
        match="k must be greater than or equal to 1",
    ):
        Retriever(vector_store=vector_store, k=k)

    vector_store.similarity_search.assert_not_called()


@pytest.mark.parametrize("query", ["", "   "])
def test_retriever_rejects_empty_query(
    vector_store: MagicMock,
    query: str,
) -> None:
    """Verify blank queries are rejected before retrieval."""
    retriever = Retriever(vector_store=vector_store)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve(query)

    vector_store.similarity_search.assert_not_called()


def test_retriever_returns_relevant_documents(
    vector_store: MagicMock,
) -> None:
    """Verify retrieved documents are returned unchanged."""
    expected_documents = [
        Document(
            page_content="Retrieval finds relevant information.",
            metadata={"source": "retrieval.md"},
        ),
        Document(
            page_content="Embeddings represent text as vectors.",
            metadata={"source": "embeddings.md"},
        ),
    ]
    vector_store.similarity_search.return_value = expected_documents
    retriever = Retriever(vector_store=vector_store, k=2)

    results = retriever.retrieve("How does retrieval work?")

    assert results == expected_documents
    vector_store.similarity_search.assert_called_once_with(
        query="How does retrieval work?",
        k=2,
    )
