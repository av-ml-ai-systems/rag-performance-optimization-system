# Module: test_answer_generator
# Path: tests/unit/test_answer_generator.py
# Purpose: Test the answer generation component.
# Description: Verify model initialization, configuration capture, execution
# latency, token usage extraction, streaming TTFT, and error handling.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from rag_performance.answer_generator import AnswerGenerator


@dataclass
class FakeMessage:
    """Test double for chat model responses."""

    content: Any
    usage_metadata: dict[str, Any] | None = None
    response_metadata: dict[str, Any] | None = None


def test_rejects_empty_model() -> None:
    """Reject an empty model name before initializing the client."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        with pytest.raises(ValueError, match="model must not be empty"):
            AnswerGenerator(model=" ")

        mock_chat.assert_not_called()


def test_initializes_client_with_selected_model() -> None:
    """Initialize ChatOllama with the selected model name and default temp."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        generator = AnswerGenerator(model="llama3.2:3b")

        assert generator.model_name == "llama3.2:3b"
        assert generator.temperature == 0.0
        assert generator.stream is False
        mock_chat.assert_called_once_with(
            model="llama3.2:3b",
            temperature=0.0,
        )


def test_initializes_client_with_custom_temperature() -> None:
    """Initialize ChatOllama with custom temperature configuration."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        generator = AnswerGenerator(
            model="llama3.2:3b",
            temperature=0.7,
        )

        assert generator.temperature == 0.7
        mock_chat.assert_called_once_with(
            model="llama3.2:3b",
            temperature=0.7,
        )


def test_rejects_negative_temperature() -> None:
    """Reject a negative temperature before initializing the client."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            AnswerGenerator(temperature=-0.5)

        mock_chat.assert_not_called()


def test_rejects_empty_prompt_without_invoking_model() -> None:
    """Reject an empty prompt without calling the model."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        generator = AnswerGenerator()

        with pytest.raises(ValueError, match="prompt must not be empty"):
            generator.generate(" ")

        mock_chat.return_value.invoke.assert_not_called()


def test_generates_and_strips_answer() -> None:
    """Return generated text without surrounding whitespace and track latency."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value = FakeMessage(
            content="  Generated answer.  "
        )
        generator = AnswerGenerator()

        answer = generator.generate("Explain RAG.")

        assert answer == "Generated answer."
        assert generator.last_latency_ms >= 0.0
        assert generator.last_ttft_ms is None
        mock_chat.return_value.invoke.assert_called_once_with("Explain RAG.")


def test_preserves_latency_on_generation_failure() -> None:
    """Preserve execution latency even when model invocation raises error."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.side_effect = RuntimeError(
            "Ollama connection error"
        )
        generator = AnswerGenerator()

        with pytest.raises(RuntimeError, match="Ollama connection error"):
            generator.generate("Explain RAG.")

        assert generator.last_latency_ms >= 0.0


def test_rejects_non_text_response() -> None:
    """Raise an error when the model returns non-text content."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value = FakeMessage(
            content=[{"type": "image", "url": "example"}]
        )
        generator = AnswerGenerator()

        with pytest.raises(
            TypeError,
            match="The language model returned non-text content",
        ):
            generator.generate("Describe this image.")


def test_extracts_tokens_from_usage_metadata() -> None:
    """Extract input, output, and total tokens from usage_metadata."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_response = FakeMessage(
            content="Answer with token usage.",
            usage_metadata={
                "input_tokens": 42,
                "output_tokens": 18,
                "total_tokens": 60,
            },
        )
        mock_chat.return_value.invoke.return_value = mock_response

        generator = AnswerGenerator()
        generator.generate("Prompt query.")

        usage = generator.last_token_usage
        assert usage.prompt_tokens == 42
        assert usage.generation_tokens == 18
        assert usage.total_tokens == 60


def test_extracts_tokens_from_response_metadata_fallback() -> None:
    """Extract Ollama native token counts when usage_metadata is absent."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_response = FakeMessage(
            content="Answer with native Ollama metadata.",
            response_metadata={
                "prompt_eval_count": 35,
                "eval_count": 15,
            },
        )
        mock_chat.return_value.invoke.return_value = mock_response

        generator = AnswerGenerator()
        generator.generate("Prompt query.")

        usage = generator.last_token_usage
        assert usage.prompt_tokens == 35
        assert usage.generation_tokens == 15
        assert usage.total_tokens == 50


def test_token_usage_defaults_to_none_when_metadata_missing() -> None:
    """Ensure token counts default to None when metadata is unavailable."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_response = FakeMessage(
            content="Answer without token metadata.",
        )
        mock_chat.return_value.invoke.return_value = mock_response

        generator = AnswerGenerator()
        generator.generate("Prompt query.")

        usage = generator.last_token_usage
        assert usage.prompt_tokens is None
        assert usage.generation_tokens is None
        assert usage.total_tokens is None


def test_streaming_generation_measures_ttft() -> None:
    """Measure TTFT and accumulate text during streaming generation."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        chunk1 = FakeMessage(content="First ")
        chunk2 = FakeMessage(content="chunk.")
        mock_chat.return_value.stream.return_value = [chunk1, chunk2]

        generator = AnswerGenerator(stream=True)
        answer = generator.generate("Stream prompt.")

        assert answer == "First chunk."
        assert generator.last_ttft_ms is not None
        assert generator.last_ttft_ms >= 0.0
        assert generator.last_latency_ms >= 0.0
        mock_chat.return_value.stream.assert_called_once_with("Stream prompt.")


def test_blocking_generation_leaves_ttft_none() -> None:
    """Ensure TTFT is None when generation operates in blocking mode."""
    with patch("rag_performance.answer_generator.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value = FakeMessage(
            content="Blocking answer."
        )
        generator = AnswerGenerator(stream=False)
        generator.generate("Prompt.")

        assert generator.last_ttft_ms is None
