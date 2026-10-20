# Module: ingestion
# Path: src/rag_performance/ingestion.py
# Purpose: Load source documents for the RAG ingestion pipeline.
# Description: Load UTF-8 TXT files into LangChain Document objects while
# preserving source metadata and validating input paths and file contents.

from pathlib import Path

from langchain_core.documents import Document


class DocumentLoader:
    """Load UTF-8 text files into LangChain Document objects."""

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load one text file and preserve its source path as metadata."""
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.suffix.lower() != ".txt":
            raise ValueError(f"Unsupported file type: {path.suffix}")

        content = path.read_text(encoding="utf-8")

        if not content.strip():
            raise ValueError(f"Document is empty: {path}")

        return [
            Document(
                page_content=content,
                metadata={"source": str(path.resolve())},
            )
        ]

    def load_directory(self, directory_path: str | Path) -> list[Document]:
        """Load TXT files from a directory, excluding subdirectories."""
        directory = Path(directory_path)

        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        documents: list[Document] = []

        for path in sorted(directory.glob("*.txt")):
            documents.extend(self.load_file(path))

        return documents
