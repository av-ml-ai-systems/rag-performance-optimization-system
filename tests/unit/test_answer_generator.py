# Module: test_answer_generator
# Path: tests/test_answer_generator.py
# Purpose: Test the answer generation component.
# Description: Verify model initialization, input validation, generated
# answer handling, and rejection of non-text model responses.

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from rag_performance.answer_generator import AnswerGenerator


def test_rejects_empty_model() -> None:
    """Reject an empty model name before initializing the client."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        with pytest.raises(ValueError, match="model must not be empty"):
            AnswerGenerator(model=" ")

        mock_chat.assert_not_called()


def test_initializes_client_with_selected_model() -> None:
    """Initialize ChatOllama with the selected model name."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        AnswerGenerator(model="llama3.2:3b")

        mock_chat.assert_called_once_with(model="llama3.2:3b")


def test_rejects_empty_prompt_without_invoking_model() -> None:
    """Reject an empty prompt without calling the model."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        generator = AnswerGenerator()

        with pytest.raises(ValueError, match="prompt must not be empty"):
            generator.generate(" ")

        mock_chat.return_value.invoke.assert_not_called()


def test_generates_and_strips_answer() -> None:
    """Return generated text without surrounding whitespace."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value = AIMessage(
            content="  Generated answer.  "
        )
        generator = AnswerGenerator()

        answer = generator.generate("Explain RAG.")

        assert answer == "Generated answer."
        mock_chat.return_value.invoke.assert_called_once_with("Explain RAG.")


def test_rejects_non_text_response() -> None:
    """Raise an error when the model returns non-text content."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value = AIMessage(
            content=[{"type": "image", "url": "example"}]
        )
        generator = AnswerGenerator()

        with pytest.raises(
            TypeError,
            match="The language model returned non-text content",
        ):
            generator.generate("Describe this image.")
