# Module: test_prompt_builder
# Path: tests/test_prompt_builder.py
# Purpose: Validate prompt construction and input validation.
# Description: Ensure prompts include the supplied context and question,
# preserve source information, and reject empty or blank inputs.

import pytest

from rag_performance.prompt_builder import PromptBuilder


@pytest.fixture
def prompt_builder() -> PromptBuilder:
    """Provide a PromptBuilder instance."""
    return PromptBuilder()


def test_build_includes_context_and_question(
    prompt_builder: PromptBuilder,
) -> None:
    """Verify the generated prompt contains both required inputs."""
    context = "[Document 1 | Source: guide.md]\nRAG retrieves relevant documents."
    query = "How does RAG retrieve information?"

    result = prompt_builder.build(query=query, context=context)

    assert context in result
    assert query in result
    assert result.endswith("Answer:")


def test_build_includes_grounding_instructions(
    prompt_builder: PromptBuilder,
) -> None:
    """Verify the prompt instructs the model to ground its answer."""
    result = prompt_builder.build(
        query="What is RAG?",
        context="RAG combines retrieval with generation.",
    )

    assert "provided context" in result
    assert "Do not invent facts" in result
    assert "say so clearly" in result


def test_build_preserves_source_information(
    prompt_builder: PromptBuilder,
) -> None:
    """Verify source attribution survives prompt construction."""
    context = (
        "[Document 1 | Source: employee_benefits.md]\n"
        "Employees can access health benefits."
    )

    result = prompt_builder.build(
        query="What benefits are available?",
        context=context,
    )

    assert "employee_benefits.md" in result


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_build_rejects_empty_query(
    prompt_builder: PromptBuilder,
    query: str,
) -> None:
    """Verify blank questions are rejected."""
    with pytest.raises(ValueError, match="query must not be empty"):
        prompt_builder.build(
            query=query,
            context="Valid context.",
        )


@pytest.mark.parametrize("context", ["", "   ", "\n\t"])
def test_build_rejects_empty_context(
    prompt_builder: PromptBuilder,
    context: str,
) -> None:
    """Verify blank contexts are rejected."""
    with pytest.raises(ValueError, match="context must not be empty"):
        prompt_builder.build(
            query="Valid question?",
            context=context,
        )


def test_build_strips_surrounding_whitespace(
    prompt_builder: PromptBuilder,
) -> None:
    """Verify surrounding whitespace is removed from both inputs."""
    result = prompt_builder.build(
        query="  What is RAG?  ",
        context="  Retrieved context.  ",
    )

    assert "Question:\nWhat is RAG?\n\nAnswer:" in result
    assert "Context:\nRetrieved context.\n\nQuestion:" in result
