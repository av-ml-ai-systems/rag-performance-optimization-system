# Module: test_context_builder
# Path: tests/test_context_builder.py
# Purpose: Validate ContextBuilder document formatting and edge cases.
# Description: Test context construction, source attribution, document order,
# and predictable handling of empty or blank document content.

import pytest
from langchain_core.documents import Document

from rag_performance.context_builder import ContextBuilder


@pytest.fixture
def context_builder() -> ContextBuilder:
    """Provide a ContextBuilder instance."""
    return ContextBuilder()


def test_build_returns_empty_string_for_no_documents(
    context_builder: ContextBuilder,
) -> None:
    """Verify empty input produces an empty context."""
    assert context_builder.build([]) == ""


def test_build_formats_single_document_with_source(
    context_builder: ContextBuilder,
) -> None:
    """Verify document content and source are included."""
    documents = [
        Document(
            page_content="RAG retrieves relevant information.",
            metadata={"source": "rag_guide.md"},
        )
    ]

    result = context_builder.build(documents)

    assert result == (
        "[Document 1 | Source: rag_guide.md]\nRAG retrieves relevant information."
    )


def test_build_formats_multiple_documents_in_order(
    context_builder: ContextBuilder,
) -> None:
    """Verify document order and section separation are preserved."""
    documents = [
        Document(
            page_content="First document content.",
            metadata={"source": "first.md"},
        ),
        Document(
            page_content="Second document content.",
            metadata={"source": "second.md"},
        ),
    ]

    result = context_builder.build(documents)

    assert result == (
        "[Document 1 | Source: first.md]\n"
        "First document content.\n\n"
        "[Document 2 | Source: second.md]\n"
        "Second document content."
    )


def test_build_uses_fallback_when_source_is_missing(
    context_builder: ContextBuilder,
) -> None:
    """Verify documents without source metadata remain usable."""
    documents = [Document(page_content="Content without source metadata.")]

    result = context_builder.build(documents)

    assert result == (
        "[Document 1 | Source: Unknown source]\nContent without source metadata."
    )


def test_build_strips_surrounding_content_whitespace(
    context_builder: ContextBuilder,
) -> None:
    """Verify surrounding whitespace is removed from document content."""
    documents = [
        Document(
            page_content="  \n  Relevant information.  \n ",
            metadata={"source": "guide.md"},
        )
    ]

    result = context_builder.build(documents)

    assert result.endswith("\nRelevant information.")


def test_build_skips_blank_document_content(
    context_builder: ContextBuilder,
) -> None:
    """Verify blank chunks are skipped without removing later documents."""
    documents = [
        Document(
            page_content="   ",
            metadata={"source": "blank.md"},
        ),
        Document(
            page_content="Useful content.",
            metadata={"source": "useful.md"},
        ),
    ]

    result = context_builder.build(documents)

    assert result == ("[Document 2 | Source: useful.md]\nUseful content.")


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_build_returns_empty_string_for_only_blank_documents(
    context_builder: ContextBuilder,
    content: str,
) -> None:
    """Verify blank-only input produces no context sections."""
    documents = [Document(page_content=content)]

    assert context_builder.build(documents) == ""
