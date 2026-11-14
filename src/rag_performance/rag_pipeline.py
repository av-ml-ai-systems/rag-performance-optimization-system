# Module: rag_pipeline
# Path: src/rag_performance/rag_pipeline.py
# Purpose: Orchestrate the complete RAG query pipeline.
# Description: Coordinate retrieval, context construction, prompt preparation,
# and answer generation through injected components, with request correlation
# and structured operational logging.

from __future__ import annotations

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.context_builder import ContextBuilder
from rag_performance.logging import RequestContext, default_context, get_logger
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
        context: RequestContext | None = None,
    ) -> None:
        """Initialize the pipeline with its required components."""
        self._retriever = retriever
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._answer_generator = answer_generator
        self._context = context or default_context
        self._logger = get_logger(__name__, context=self._context)

    def query(self, question: str, request_id: str | None = None) -> str:
        """Retrieve relevant context and generate an answer."""
        with self._context.scope(request_id):
            self._logger.info("RAG query pipeline execution started")

            if not question.strip():
                self._logger.warning("Query rejected: question must not be empty")
                raise ValueError("question must not be empty.")

            try:
                documents = self._retriever.retrieve(question)
                self._logger.info(
                    "Retrieval completed (documents_retrieved=%d)",
                    len(documents),
                )

                context = self._context_builder.build(documents)
                if not context.strip():
                    self._logger.warning("Context construction yielded empty context")
                    raise ValueError("No usable context was retrieved.")

                self._logger.info(
                    "Context construction completed (context_size_chars=%d)",
                    len(context),
                )

                prompt = self._prompt_builder.build(
                    query=question,
                    context=context,
                )
                self._logger.info(
                    "Prompt construction completed (prompt_size_chars=%d)",
                    len(prompt),
                )

                answer = self._answer_generator.generate(prompt)
                self._logger.info("Answer generation completed successfully")
                return answer

            except Exception as exc:
                if not isinstance(exc, ValueError):
                    self._logger.error(
                        "Unhandled failure during pipeline execution: %s",
                        type(exc).__name__,
                        exc_info=True,
                    )
                raise
