# Module: test_answer_generator_integration
# Path: tests/integration/test_answer_generator_integration.py
# Purpose: Verify answer generation against a real Ollama model.
# Description: Exercise AnswerGenerator without mocking ChatOllama,
# confirming that the local model returns a usable text response.

import pytest

from rag_performance.answer_generator import AnswerGenerator


@pytest.mark.integration
def test_generates_answer_with_real_ollama_model() -> None:
    """Generate a non-empty answer using the configured local model."""
    generator = AnswerGenerator(model="llama3.2:3b")

    answer = generator.generate(
        "In one sentence, explain what retrieval-augmented generation (RAG) is."
    )

    assert isinstance(answer, str)
    assert answer.strip()
