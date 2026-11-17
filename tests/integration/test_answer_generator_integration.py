# Module: test_answer_generator_integration
# Path: tests/integration/test_answer_generator_integration.py
# Purpose: Validate AnswerGenerator integration with Ollama.
# Description: Verify that AnswerGenerator connects to the local Ollama model,
# extracts real token usage, and measures monotonic latency and streaming TTFT.

from __future__ import annotations

import pytest

from rag_performance.answer_generator import AnswerGenerator


@pytest.mark.integration
def test_answer_generator_returns_non_empty_answer() -> None:
    """Verify AnswerGenerator generates a real response and extracts tokens."""
    generator = AnswerGenerator()

    answer = generator.generate(
        "Context:\n"
        "The standard working schedule is Monday to Friday.\n\n"
        "Question:\n"
        "What is the standard working schedule?"
    )

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert "monday" in answer.lower()
    assert "friday" in answer.lower()

    assert generator.last_latency_ms > 0.0
    assert generator.last_ttft_ms is None

    usage = generator.last_token_usage
    assert usage.prompt_tokens is not None
    assert usage.prompt_tokens > 0
    assert usage.generation_tokens is not None
    assert usage.generation_tokens > 0
    assert usage.total_tokens == (usage.prompt_tokens + usage.generation_tokens)


@pytest.mark.integration
def test_answer_generator_streaming_measures_ttft() -> None:
    """Verify streaming mode measures positive TTFT from live Ollama."""
    generator = AnswerGenerator(stream=True)

    answer = generator.generate("Reply with the single word OK.")

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert generator.last_ttft_ms is not None
    assert generator.last_ttft_ms > 0.0
    assert generator.last_latency_ms >= generator.last_ttft_ms
