# Module: test_rag_pipeline
# Path: tests/unit/test_rag_pipeline.py
# Purpose: Validate RAG pipeline orchestration in isolation.
# Description: Verify component call order, generated answers, input validation,
# and behavior when retrieval produces no usable context.

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.context_builder import ContextBuilder
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
