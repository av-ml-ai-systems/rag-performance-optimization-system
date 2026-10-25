# Module: test_ingestion_chunking_integration
# Path: tests/test_ingestion_chunking_integration.py
# Purpose: Validate the integration between document ingestion and chunking.
# Description: Load the real knowledge base, split documents into chunks,
# and verify content, source metadata, and source-file coverage.

from pathlib import Path

from rag_performance.chunking import TextChunker
from rag_performance.ingestion import DocumentLoader


def test_ingestion_and_chunking_integration() -> None:
    """Test ingestion and chunking against the real knowledge base."""
    knowledge_base = Path("data/knowledge_base")

    documents = DocumentLoader().load_directory(knowledge_base)
    chunks = TextChunker().split_documents(documents)

    expected_sources = {
        str(path.resolve())
        for path in knowledge_base.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}
    }
    actual_sources = {
        str(chunk.metadata["source"]) for chunk in chunks if "source" in chunk.metadata
    }

    assert len(documents) == 8
    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)
    assert actual_sources == expected_sources
