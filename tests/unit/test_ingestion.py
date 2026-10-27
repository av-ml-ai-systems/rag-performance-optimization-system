# Module: test_ingestion
# Path: tests/test_ingestion.py
# Purpose: Validate the DocumentLoader implementation.
# Description: Test TXT, Markdown, and PDF loading, source metadata,
# input validation, and deterministic directory loading.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_performance.ingestion import DocumentLoader


def test_load_file_returns_document_with_source_metadata(
    tmp_path: Path,
) -> None:
    """Test loading a TXT file and preserving its source metadata."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "RAG improves information retrieval.",
        encoding="utf-8",
    )

    documents = DocumentLoader().load_file(file_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == ("RAG improves information retrieval.")
    assert documents[0].metadata["source"] == str(file_path.resolve())


def test_load_file_loads_markdown(tmp_path: Path) -> None:
    """Test loading a Markdown file without changing its content."""
    file_path = tmp_path / "sample.md"
    content = "# RAG Policy\n\nUse relevant context."
    file_path.write_text(content, encoding="utf-8")

    documents = DocumentLoader().load_file(file_path)

    assert len(documents) == 1
    assert documents[0].page_content == content
    assert documents[0].metadata["source"] == str(file_path.resolve())


def test_load_file_extracts_pdf_text(tmp_path: Path) -> None:
    """Test combining extracted text from all PDF pages."""
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"Mock PDF fixture")

    first_page = MagicMock()
    first_page.extract_text.return_value = "First page content."

    second_page = MagicMock()
    second_page.extract_text.return_value = "Second page content."

    reader = MagicMock()
    reader.pages = [first_page, second_page]

    with patch(
        "rag_performance.ingestion.PdfReader",
        return_value=reader,
    ):
        documents = DocumentLoader().load_file(file_path)

    assert len(documents) == 1
    assert documents[0].page_content == ("First page content.\nSecond page content.")
    assert documents[0].metadata["source"] == str(file_path.resolve())


def test_load_file_raises_for_missing_file(tmp_path: Path) -> None:
    """Test that loading a nonexistent file raises FileNotFoundError."""
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Document not found"):
        DocumentLoader().load_file(file_path)


def test_load_file_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Test that unsupported file extensions are rejected."""
    file_path = tmp_path / "sample.csv"
    file_path.write_text("Example content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load_file(file_path)


def test_load_file_rejects_empty_content(tmp_path: Path) -> None:
    """Test that an empty text file is rejected."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="Document is empty"):
        DocumentLoader().load_file(file_path)


def test_load_directory_returns_supported_documents(
    tmp_path: Path,
) -> None:
    """Test loading supported files in deterministic filename order."""
    (tmp_path / "b.txt").write_text(
        "Second document",
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text(
        "First document",
        encoding="utf-8",
    )
    (tmp_path / "ignored.csv").write_text(
        "Ignore this",
        encoding="utf-8",
    )

    documents = DocumentLoader().load_directory(tmp_path)

    assert len(documents) == 2
    assert [document.page_content for document in documents] == [
        "First document",
        "Second document",
    ]


def test_load_directory_raises_for_missing_directory(
    tmp_path: Path,
) -> None:
    """Test that a nonexistent directory raises NotADirectoryError."""
    directory = tmp_path / "missing"

    with pytest.raises(NotADirectoryError, match="Directory not found"):
        DocumentLoader().load_directory(directory)
