# Module: test_rag_pipeline_integration
# Path: tests/integration/test_rag_pipeline_integration.py
# Purpose: Validate the complete RAG query pipeline with real local components.
# Description: Load and chunk the knowledge base, generate embeddings,
# persist vectors in a temporary ChromaDB collection, retrieve relevant context,
# generate an answer via Ollama, and validate end-to-end observability.

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import pytest

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.chunking import TextChunker
from rag_performance.context_builder import ContextBuilder
from rag_performance.embeddings import OllamaEmbeddingProvider
from rag_performance.ingestion import DocumentLoader
from rag_performance.metrics_collector import MetricsCollector
from rag_performance.metrics_reporting import MetricsAggregator
from rag_performance.metrics_store import JsonlMetricsStore
from rag_performance.prompt_builder import PromptBuilder
from rag_performance.rag_pipeline import RAGPipeline
from rag_performance.retriever import Retriever
from rag_performance.vector_store import VectorStore


@pytest.mark.integration
def test_end_to_end_rag_pipeline(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify end-to-end RAG answer generation and full observability."""
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

    store = JsonlMetricsStore(tmp_path / "telemetry.jsonl")
    collector = MetricsCollector(store=store)
    question = "What employee benefits are described in the knowledge base?"
    trace_id = "e2e-observability-001"

    with caplog.at_level(logging.INFO):
        answer = collector.run_and_collect(
            pipeline=pipeline,
            question=question,
            request_id=trace_id,
        )

    # 1. Verify functional RAG answer
    assert isinstance(answer, str)
    assert answer.strip() != ""

    # 2. Verify request_id propagation across all 5 core component loggers
    expected_loggers = {
        "rag_performance.rag_pipeline",
        "rag_performance.retriever",
        "rag_performance.context_builder",
        "rag_performance.prompt_builder",
        "rag_performance.answer_generator",
    }
    observed_loggers = {
        rec.name for rec in caplog.records if rec.name.startswith("rag_performance.")
    }
    assert expected_loggers.issubset(observed_loggers)

    for record in caplog.records:
        if record.name in expected_loggers:
            assert getattr(record, "request_id", None) == trace_id
            assert question not in record.message
            assert answer not in record.message

    # 3. Verify RequestTelemetry consistency and accuracy
    telemetry = pipeline.last_telemetry
    assert telemetry is not None
    assert telemetry.request_id == trace_id
    assert telemetry.error.occurred is False

    assert telemetry.retrieval.k_configured == 4
    assert telemetry.retrieval.documents_retrieved == 4
    assert telemetry.retrieval.latency_ms > 0.0
    assert telemetry.retrieval.status == "success"

    assert telemetry.context.chunks_used == 4
    assert telemetry.context.context_size_chars > 0
    assert telemetry.context.context_size_tokens_estimated == (
        telemetry.context.context_size_chars // 4
    )
    assert telemetry.context.latency_ms > 0.0

    assert telemetry.prompt.latency_ms > 0.0

    assert telemetry.llm.model == "llama3.2:3b"
    assert telemetry.llm.latency_ms > 0.0
    assert telemetry.llm.ttft_ms is None
    assert telemetry.llm.prompt_tokens is not None
    assert telemetry.llm.prompt_tokens > 0
    assert telemetry.llm.generation_tokens is not None
    assert telemetry.llm.generation_tokens > 0
    assert telemetry.llm.total_tokens == (
        telemetry.llm.prompt_tokens + telemetry.llm.generation_tokens
    )

    stage_sum = (
        telemetry.retrieval.latency_ms
        + telemetry.context.latency_ms
        + telemetry.prompt.latency_ms
        + telemetry.llm.latency_ms
    )
    assert telemetry.total_latency_ms >= stage_sum

    # 4. Verify persistence and aggregation consistency
    persisted = collector.get_all()
    assert len(persisted) == 1
    assert persisted[0]["request_id"] == trace_id

    report = MetricsAggregator().aggregate(persisted)
    assert report.total_requests == 1
    assert report.successful_requests == 1
    assert report.failed_requests == 0
    assert report.total_latency_ms.mean == pytest.approx(telemetry.total_latency_ms)
