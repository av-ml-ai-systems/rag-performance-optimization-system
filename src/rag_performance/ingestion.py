# Module: ingestion
# Path: src/rag_performance/ingestion.py
# Purpose: Load source documents for the RAG ingestion pipeline.
# Description: Load UTF-8 TXT and Markdown files and extract text from PDFs
# into LangChain Document objects while preserving source metadata.

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


class DocumentLoader:
    """Load TXT, Markdown, and PDF files into LangChain Documents."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load one supported file and preserve its source path."""
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix.lower() == ".pdf":
            content = self._load_pdf(path)
        else:
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
        """Load supported files from a directory in deterministic order."""
        directory = Path(directory_path)

        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        documents: list[Document] = []

        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        for path in paths:
            documents.extend(self.load_file(path))

        return documents

    @staticmethod
    def _load_pdf(path: Path) -> str:
        """Extract text from all pages of a PDF."""
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
