# Module: test_rag_pipeline
# Path: tests/unit/test_rag_pipeline.py
# Purpose: Validate RAG pipeline orchestration in isolation.
# Description: Verify component call order, generated answers, input validation,
# behavior when retrieval produces no usable context, request_id context lifecycle,
# and end-to-end structured log correlation across components.

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


@pytest.fixture
def components() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Provide mocked components for isolated pipeline tests."""
    retriever = MagicMock(spec=Retriever)
    context_builder = MagicMock(spec=ContextBuilder)
    prompt_builder = MagicMock(spec=PromptBuilder)
    answer_generator = MagicMock(spec=AnswerGenerator)

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
    """Verify explicit request_id is active during execution and cleared on exit."""
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
    """Verify omitted request_id generates UUID hex in context and clears on exit."""
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
    """Verify request_id is cleared even when validation rejects empty question."""
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
    """Verify request_id is cleared if a component throws an unexpected exception."""
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

    # Verify data privacy: raw query text must never appear in log messages
    for record in pipeline_records:
        assert "Sample query" not in record.message
