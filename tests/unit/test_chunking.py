# Module: test_chunking
# Path: tests/test_chunking.py
# Purpose: Validate document chunking behavior and input validation.
# Description: Test chunk sizes, overlap, metadata preservation,
# empty inputs, and invalid configuration.

import pytest
from langchain_core.documents import Document

from rag_performance.chunking import TextChunker


def test_split_documents_splits_long_document() -> None:
    """Test that long documents are divided into smaller chunks."""
    document = Document(
        page_content=" ".join(["retrieval"] * 100),
        metadata={"source": "sample.txt"},
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.split_documents([document])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_split_documents_preserves_metadata() -> None:
    """Test that source metadata is retained in every chunk."""
    document = Document(
        page_content=" ".join(["information"] * 80),
        metadata={"source": "policy.md", "category": "HR"},
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.split_documents([document])

    assert chunks
    assert all(chunk.metadata == document.metadata for chunk in chunks)


def test_split_documents_returns_empty_list_for_empty_input() -> None:
    """Test that an empty document list returns an empty list."""
    assert TextChunker().split_documents([]) == []


def test_split_documents_rejects_empty_document() -> None:
    """Test that documents containing only whitespace are rejected."""
    document = Document(page_content="   \n\t")

    with pytest.raises(
        ValueError,
        match="Input documents must not contain empty text",
    ):
        TextChunker().split_documents([document])


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_init_rejects_invalid_configuration(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Test that invalid chunk sizes and overlaps are rejected."""
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_split_documents_preserves_source_document_content() -> None:
    """Test that splitting retains the original document's text."""
    content = " ".join(["semantic search"] * 60)
    document = Document(page_content=content)
    chunks = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    ).split_documents([document])

    reconstructed = "".join(chunk.page_content for chunk in chunks)

    for chunk in chunks:
        assert chunk.page_content in content

    assert reconstructed
