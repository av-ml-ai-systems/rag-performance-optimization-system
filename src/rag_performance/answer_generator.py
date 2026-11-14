# Module: answer_generator
# Path: src/rag_performance/answer_generator.py
# Purpose: Generate natural-language answers using a local Ollama model.
# Description: Encapsulate LangChain's Ollama integration behind a focused
# component that accepts a completed prompt and returns the generated answer.

from __future__ import annotations

from langchain_ollama import ChatOllama

from rag_performance.logging import get_logger

logger = get_logger(__name__)


class AnswerGenerator:
    """Generate answers using a locally hosted language model."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
    ) -> None:
        """Initialize the generator with the selected Ollama model."""
        if not model.strip():
            logger.warning("AnswerGenerator initialization rejected: model is empty")
            raise ValueError("model must not be empty.")

        self._model_name = model.strip()
        self._model = ChatOllama(model=self._model_name)

    def generate(self, prompt: str) -> str:
        """Generate an answer from a non-empty prompt."""
        if not prompt.strip():
            logger.warning("Generation rejected: prompt must not be empty")
            raise ValueError("prompt must not be empty.")

        logger.info(
            "LLM generation started (model=%s, prompt_size_chars=%d)",
            self._model_name,
            len(prompt),
        )

        try:
            response = self._model.invoke(prompt)
        except Exception as exc:
            logger.error(
                "LLM generation failed: %s",
                type(exc).__name__,
                exc_info=True,
            )
            raise

        answer = response.content

        if not isinstance(answer, str):
            logger.error(
                "LLM returned unexpected content type: %s",
                type(answer).__name__,
            )
            raise TypeError("The language model returned non-text content.")

        resolved_answer = answer.strip()
        logger.info(
            "LLM generation completed successfully (answer_size_chars=%d)",
            len(resolved_answer),
        )

        return resolved_answer
