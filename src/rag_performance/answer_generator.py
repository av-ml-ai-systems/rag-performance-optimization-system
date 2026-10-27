# Module: answer_generator
# Path: src/rag_performance/answer_generator.py
# Purpose: Generate natural-language answers using a local Ollama model.
# Description: Encapsulate LangChain's Ollama integration behind a focused
# component that accepts a completed prompt and returns the generated answer.

from langchain_ollama import ChatOllama


class AnswerGenerator:
    """Generate answers using a locally hosted language model."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
    ) -> None:
        """Initialize the generator with the selected Ollama model."""
        if not model.strip():
            raise ValueError("model must not be empty.")

        self._model = ChatOllama(model=model)

    def generate(self, prompt: str) -> str:
        """Generate an answer from a non-empty prompt."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        response = self._model.invoke(prompt)
        answer = response.content

        if not isinstance(answer, str):
            raise TypeError("The language model returned non-text content.")

        return answer.strip()
