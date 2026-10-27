# Module: rag_pipeline
# Path: src/rag_performance/rag_pipeline.py
# Purpose: Orchestrate the complete RAG query pipeline.
# Description: Coordinate retrieval, context construction, prompt preparation,
# and answer generation through injected components.

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.context_builder import ContextBuilder
from rag_performance.prompt_builder import PromptBuilder
from rag_performance.retriever import Retriever


class RAGPipeline:
    """Coordinate the components required to answer a RAG query."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        answer_generator: AnswerGenerator,
    ) -> None:
        """Initialize the pipeline with its required components."""
        self._retriever = retriever
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._answer_generator = answer_generator

    def query(self, question: str) -> str:
        """Retrieve relevant context and generate an answer."""
        if not question.strip():
            raise ValueError("question must not be empty.")

        documents = self._retriever.retrieve(question)
        context = self._context_builder.build(documents)

        if not context.strip():
            raise ValueError("No usable context was retrieved.")

        prompt = self._prompt_builder.build(
            query=question,
            context=context,
        )

        return self._answer_generator.generate(prompt)
