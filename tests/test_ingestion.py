# Module: test_ingestion
# Path: tests/test_ingestion.py
# Purpose: Validate the DocumentLoader implementation.
# Description: Test text-file loading, source metadata, input validation,
# and directory loading using temporary files.

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_performance.ingestion import DocumentLoader


def test_load_file_returns_document_with_source_metadata(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("RAG improves information retrieval.", encoding="utf-8")

    documents = DocumentLoader().load_file(file_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "RAG improves information retrieval."
    assert documents[0].metadata["source"] == str(file_path.resolve())


def test_load_file_raises_for_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        DocumentLoader().load_file(file_path)


def test_load_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_text("Example content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load_file(file_path)


def test_load_file_rejects_empty_content(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="Document is empty"):
        DocumentLoader().load_file(file_path)


def test_load_directory_returns_all_txt_documents(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("Second document", encoding="utf-8")
    (tmp_path / "a.txt").write_text("First document", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("Ignore this", encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)

    assert len(documents) == 2
    assert [document.page_content for document in documents] == [
        "First document",
        "Second document",
    ]


def test_load_directory_raises_for_missing_directory(tmp_path: Path) -> None:
    directory = tmp_path / "missing"

    with pytest.raises(NotADirectoryError):
        DocumentLoader().load_directory(directory)
