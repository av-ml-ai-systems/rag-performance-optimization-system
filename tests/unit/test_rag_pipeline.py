# Module: test_rag_pipeline
# Path: tests/unit/test_rag_pipeline.py
# Purpose: Validate RAG pipeline orchestration in isolation.
# Description: Verify component call order, generated answers, input validation,
# behavior when retrieval produces no usable context, request_id context
# lifecycle, end-to-end log correlation, and consolidated telemetry capture.

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.context_builder import ContextBuilder
from rag_performance.logging import RequestContext
from rag_performance.prompt_builder import PromptBuilder
from rag_performance.rag_pipeline import RAGPipeline
from rag_performance.retriever import Retriever
from rag_performance.telemetry import TokenUsage


@pytest.fixture
def components() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Provide mocked components for isolated pipeline tests."""
    retriever = MagicMock(spec=Retriever)
    retriever.k = 4
    context_builder = MagicMock(spec=ContextBuilder)
    context_builder.last_chunks_used = 1
    prompt_builder = MagicMock(spec=PromptBuilder)
    answer_generator = MagicMock(spec=AnswerGenerator)
    answer_generator.model_name = "llama3.2:3b"
    answer_generator.last_ttft_ms = None
    answer_generator.last_token_usage = TokenUsage()

    return retriever, context_builder, prompt_builder, answer_generator


@pytest.fixture
def pipeline(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> RAGPipeline:
    """Create a pipeline with mocked dependencies."""
    retriever, context_builder, prompt_builder, answer_generator = components

    return RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
    )


def test_query_orchestrates_components(
    pipeline: RAGPipeline,
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify a question flows through each component."""
    retriever, context_builder, prompt_builder, answer_generator = components
    documents = [Document(page_content="RAG uses retrieved context.")]
    context = "[Document 1 | Source: Unknown source]\nRAG uses retrieved context."
    prompt = "Prepared RAG prompt."
    answer = "RAG combines retrieval and generation."

    retriever.retrieve.return_value = documents
    context_builder.build.return_value = context
    prompt_builder.build.return_value = prompt
    answer_generator.generate.return_value = answer

    result = pipeline.query("What is RAG?")

    assert result == answer
    retriever.retrieve.assert_called_once_with("What is RAG?")
    context_builder.build.assert_called_once_with(documents)
    prompt_builder.build.assert_called_once_with(
        query="What is RAG?",
        context=context,
    )
    answer_generator.generate.assert_called_once_with(prompt)


@pytest.mark.parametrize("question", ["", "   "])
def test_query_rejects_empty_question(
    pipeline: RAGPipeline,
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    question: str,
) -> None:
    """Verify empty questions are rejected before retrieval."""
    retriever, _, _, _ = components

    with pytest.raises(ValueError, match="question must not be empty"):
        pipeline.query(question)

    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize("context", ["", "   "])
def test_query_rejects_empty_context(
    pipeline: RAGPipeline,
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    context: str,
) -> None:
    """Verify unusable context prevents prompt construction."""
    retriever, context_builder, prompt_builder, answer_generator = components
    retriever.retrieve.return_value = [Document(page_content="Retrieved document.")]
    context_builder.build.return_value = context

    with pytest.raises(ValueError, match="No usable context was retrieved"):
        pipeline.query("What is RAG?")

    prompt_builder.build.assert_not_called()
    answer_generator.generate.assert_not_called()


def test_query_rejects_empty_retrieval_results(
    pipeline: RAGPipeline,
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify an empty retrieval result cannot reach generation."""
    retriever, context_builder, prompt_builder, answer_generator = components
    retriever.retrieve.return_value = []
    context_builder.build.return_value = ""

    with pytest.raises(ValueError, match="No usable context was retrieved"):
        pipeline.query("What is RAG?")

    context_builder.build.assert_called_once_with([])
    prompt_builder.build.assert_not_called()
    answer_generator.generate.assert_not_called()


def test_query_propagates_custom_request_id(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify explicit request_id is active and cleared on exit."""
    retriever, context_builder, prompt_builder, answer_generator = components
    custom_context = RequestContext()
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        context=custom_context,
    )

    captured_id: list[str | None] = []

    def mock_retrieve(_query: str) -> list[Document]:
        captured_id.append(custom_context.get_request_id())
        return [Document(page_content="Content")]

    retriever.retrieve.side_effect = mock_retrieve
    context_builder.build.return_value = "Usable context"
    prompt_builder.build.return_value = "Prompt"
    answer_generator.generate.return_value = "Answer"

    result = pipeline.query("Question", request_id="custom-req-001")

    assert result == "Answer"
    assert captured_id == ["custom-req-001"]
    assert custom_context.get_request_id() is None


def test_query_auto_generates_request_id(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify omitted request_id generates UUID hex and clears on exit."""
    retriever, context_builder, prompt_builder, answer_generator = components
    custom_context = RequestContext()
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        context=custom_context,
    )

    captured_id: list[str | None] = []

    def mock_retrieve(_query: str) -> list[Document]:
        captured_id.append(custom_context.get_request_id())
        return [Document(page_content="Content")]

    retriever.retrieve.side_effect = mock_retrieve
    context_builder.build.return_value = "Usable context"
    prompt_builder.build.return_value = "Prompt"
    answer_generator.generate.return_value = "Answer"

    pipeline.query("Question")

    assert len(captured_id) == 1
    assert isinstance(captured_id[0], str)
    assert len(captured_id[0]) == 32
    assert custom_context.get_request_id() is None


def test_query_cleans_up_context_on_validation_failure(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify request_id is cleared when validation rejects empty question."""
    retriever, context_builder, prompt_builder, answer_generator = components
    custom_context = RequestContext()
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        context=custom_context,
    )

    with pytest.raises(ValueError, match="question must not be empty"):
        pipeline.query("", request_id="failed-validation-id")

    assert custom_context.get_request_id() is None


def test_query_cleans_up_context_on_unexpected_failure(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify request_id is cleared when an unexpected exception occurs."""
    retriever, context_builder, prompt_builder, answer_generator = components
    custom_context = RequestContext()
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        context=custom_context,
    )

    retriever.retrieve.side_effect = RuntimeError("Database connection timed out")

    with pytest.raises(RuntimeError, match="Database connection timed out"):
        pipeline.query("Question", request_id="failed-execution-id")

    assert custom_context.get_request_id() is None


def test_query_emits_structured_logs_with_request_id(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify all pipeline lifecycle logs contain the active request_id."""
    retriever, context_builder, prompt_builder, answer_generator = components
    custom_context = RequestContext()
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        context=custom_context,
    )

    retriever.retrieve.return_value = [Document(page_content="Valid content")]
    context_builder.build.return_value = "Formatted context"
    prompt_builder.build.return_value = "Assembled prompt"
    answer_generator.generate.return_value = "Generated response"

    with caplog.at_level(logging.INFO):
        pipeline.query("Sample query", request_id="traceable-id-999")

    pipeline_records = [
        rec for rec in caplog.records if rec.name == "rag_performance.rag_pipeline"
    ]
    assert len(pipeline_records) >= 5

    for record in pipeline_records:
        assert getattr(record, "request_id", None) == "traceable-id-999"

    for record in pipeline_records:
        assert "Sample query" not in record.message


def test_query_populates_consolidated_telemetry(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify successful query populates complete telemetry record."""
    retriever, context_builder, prompt_builder, answer_generator = components
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
    )

    retriever.retrieve.return_value = [Document(page_content="Context text.")]
    context_builder.build.return_value = "Formatted context content."
    context_builder.last_chunks_used = 1
    prompt_builder.build.return_value = "Prepared prompt."
    answer_generator.generate.return_value = "Final answer."
    answer_generator.last_ttft_ms = 125.5
    answer_generator.last_token_usage = TokenUsage(
        prompt_tokens=100,
        generation_tokens=50,
        total_tokens=150,
    )

    pipeline.query("What is RAG?", request_id="telemetry-req-001")

    telemetry = pipeline.last_telemetry
    assert telemetry is not None
    assert telemetry.request_id == "telemetry-req-001"
    assert telemetry.total_latency_ms >= 0.0

    # Retrieval assertions
    assert telemetry.retrieval.k_configured == 4
    assert telemetry.retrieval.documents_retrieved == 1
    assert telemetry.retrieval.status == "success"
    assert telemetry.retrieval.latency_ms >= 0.0

    # Context assertions
    assert telemetry.context.chunks_used == 1
    assert telemetry.context.context_size_chars == len("Formatted context content.")
    assert telemetry.context.context_size_tokens_estimated == (
        len("Formatted context content.") // 4
    )
    assert telemetry.context.latency_ms >= 0.0

    # Prompt & LLM assertions
    assert telemetry.prompt.latency_ms >= 0.0
    assert telemetry.llm.model == "llama3.2:3b"
    assert telemetry.llm.status == "success"
    assert telemetry.llm.latency_ms >= 0.0
    assert telemetry.llm.ttft_ms == 125.5
    assert telemetry.llm.prompt_tokens == 100
    assert telemetry.llm.generation_tokens == 50
    assert telemetry.llm.total_tokens == 150

    # Error assertions
    assert telemetry.error.occurred is False
    assert telemetry.error.stage is None

    # Serialization schema validation
    serialized = telemetry.to_dict()
    assert serialized["request_id"] == "telemetry-req-001"
    assert serialized["llm"]["ttft_ms"] == 125.5
    assert serialized["llm"]["total_tokens"] == 150


def test_query_preserves_none_for_unavailable_llm_metrics(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Ensure unavailable TTFT and tokens remain None rather than zero."""
    retriever, context_builder, prompt_builder, answer_generator = components
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
    )

    retriever.retrieve.return_value = [Document(page_content="Context text.")]
    context_builder.build.return_value = "Formatted context."
    prompt_builder.build.return_value = "Prompt."
    answer_generator.generate.return_value = "Answer."

    pipeline.query("What is RAG?", request_id="telemetry-none-001")

    telemetry = pipeline.last_telemetry
    assert telemetry is not None
    assert telemetry.llm.ttft_ms is None
    assert telemetry.llm.prompt_tokens is None
    assert telemetry.llm.generation_tokens is None
    assert telemetry.llm.total_tokens is None


def test_query_preserves_partial_telemetry_on_failure(
    components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Verify partial telemetry and error details are preserved on crash."""
    retriever, context_builder, prompt_builder, answer_generator = components
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
    )

    retriever.retrieve.return_value = [Document(page_content="Valid chunk")]
    context_builder.build.return_value = "Usable context"
    context_builder.last_chunks_used = 1
    prompt_builder.build.return_value = "Ready prompt"
    answer_generator.generate.side_effect = RuntimeError("Ollama connection refused")

    with pytest.raises(RuntimeError, match="Ollama connection refused"):
        pipeline.query("Sample question", request_id="failing-req-002")

    telemetry = pipeline.last_telemetry
    assert telemetry is not None
    assert telemetry.request_id == "failing-req-002"

    # Prior stages must be preserved
    assert telemetry.retrieval.status == "success"
    assert telemetry.retrieval.documents_retrieved == 1
    assert telemetry.context.chunks_used == 1
    assert telemetry.context.context_size_chars == len("Usable context")

    # Failing stage must record failure
    assert telemetry.llm.status == "failed"
    assert telemetry.error.occurred is True
    assert telemetry.error.stage == "llm"
    assert "Ollama connection refused" in (telemetry.error.message or "")
    assert telemetry.total_latency_ms >= 0.0
