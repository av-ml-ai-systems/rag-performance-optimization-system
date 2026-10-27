# Module: test_rag_pipeline_integration
# Path: tests/integration/test_rag_pipeline_integration.py
# Purpose: Validate the complete RAG query pipeline with real local components.
# Description: Load and chunk the knowledge base, generate embeddings,
# persist vectors in a temporary ChromaDB collection, retrieve relevant context,
# and generate a grounded answer through the local Ollama language model.

from pathlib import Path
from uuid import uuid4

import pytest

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.chunking import TextChunker
from rag_performance.context_builder import ContextBuilder
from rag_performance.embeddings import OllamaEmbeddingProvider
from rag_performance.ingestion import DocumentLoader
from rag_performance.prompt_builder import PromptBuilder
from rag_performance.rag_pipeline import RAGPipeline
from rag_performance.retriever import Retriever
from rag_performance.vector_store import VectorStore


@pytest.mark.integration
def test_end_to_end_rag_pipeline(tmp_path: Path) -> None:
    """Verify that the complete RAG pipeline generates a usable answer."""
    knowledge_base = Path("data/knowledge_base")

    documents = DocumentLoader().load_directory(knowledge_base)
    chunks = TextChunker().split_documents(documents)

    assert len(documents) == 8
    assert chunks

    embedding_provider = OllamaEmbeddingProvider(model="nomic-embed-text:latest")

    vector_store = VectorStore(
        embedding_provider=embedding_provider,
        collection_name=f"rag_e2e_{uuid4().hex}",
        persist_directory=str(tmp_path / "chroma_db"),
    )
    vector_store.add_documents(chunks)

    retriever = Retriever(vector_store=vector_store, k=4)

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        answer_generator=AnswerGenerator(model="llama3.2:3b"),
    )

    answer = pipeline.query(
        "What employee benefits are described in the knowledge base?"
    )

    assert isinstance(answer, str)
    assert answer.strip()
