# Module: rag_pipeline
# Path: src/rag_performance/rag_pipeline.py
# Purpose: Orchestrate the complete RAG query pipeline.
# Description: Coordinate retrieval, context construction, prompt preparation,
# and answer generation with request correlation, monotonic stage timing,
# operational metadata capture, token tracking, and consolidated telemetry.

from __future__ import annotations

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.context_builder import ContextBuilder
from rag_performance.logging import RequestContext, default_context, get_logger
from rag_performance.prompt_builder import PromptBuilder
from rag_performance.retriever import Retriever
from rag_performance.telemetry import (
    ComponentLatencies,
    ExecutionTimer,
    RequestTelemetry,
)


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
        self._last_latencies: ComponentLatencies | None = None
        self._last_telemetry: RequestTelemetry | None = None

    @property
    def last_latencies(self) -> ComponentLatencies | None:
        """Return latencies captured during the most recent query."""
        return self._last_latencies

    @property
    def last_telemetry(self) -> RequestTelemetry | None:
        """Return consolidated telemetry record from the most recent query."""
        return self._last_telemetry

    def query(self, question: str, request_id: str | None = None) -> str:
        """Retrieve relevant context and generate an answer."""
        telemetry = RequestTelemetry()
        latencies = ComponentLatencies()
        total_timer = ExecutionTimer()
        current_stage = "initialization"

        with self._context.scope(request_id) as active_id:
            telemetry.request_id = active_id
            self._logger.info("RAG query pipeline execution started")

            if not question.strip():
                self._logger.warning("Query rejected: question must not be empty")
                telemetry.error.occurred = True
                telemetry.error.stage = "validation"
                telemetry.error.message = "question must not be empty."
                self._last_telemetry = telemetry
                self._last_latencies = latencies
                raise ValueError("question must not be empty.")

            try:
                with total_timer:
                    # Stage 1: Retrieval
                    current_stage = "retrieval"
                    telemetry.retrieval.k_configured = self._retriever.k
                    with ExecutionTimer() as timer:
                        documents = self._retriever.retrieve(question)
                    latencies.retrieval_ms = timer.elapsed_ms
                    telemetry.retrieval.latency_ms = timer.elapsed_ms
                    telemetry.retrieval.documents_retrieved = len(documents)
                    telemetry.retrieval.status = "success"

                    self._logger.info(
                        "Retrieval completed (documents_retrieved=%d, latency_ms=%.2f)",
                        len(documents),
                        latencies.retrieval_ms,
                    )

                    # Stage 2: Context Construction
                    current_stage = "context"
                    with ExecutionTimer() as timer:
                        context = self._context_builder.build(documents)
                    latencies.context_ms = timer.elapsed_ms
                    telemetry.context.latency_ms = timer.elapsed_ms
                    telemetry.context.chunks_used = (
                        self._context_builder.last_chunks_used
                    )
                    telemetry.context.context_size_chars = len(context)
                    telemetry.context.context_size_tokens_estimated = len(context) // 4

                    if not context.strip():
                        self._logger.warning(
                            "Context construction yielded empty context"
                        )
                        raise ValueError("No usable context was retrieved.")

                    self._logger.info(
                        "Context construction completed "
                        "(context_size_chars=%d, latency_ms=%.2f)",
                        len(context),
                        latencies.context_ms,
                    )

                    # Stage 3: Prompt Construction
                    current_stage = "prompt"
                    with ExecutionTimer() as timer:
                        prompt = self._prompt_builder.build(
                            query=question,
                            context=context,
                        )
                    latencies.prompt_ms = timer.elapsed_ms
                    telemetry.prompt.latency_ms = timer.elapsed_ms

                    self._logger.info(
                        "Prompt construction completed "
                        "(prompt_size_chars=%d, latency_ms=%.2f)",
                        len(prompt),
                        latencies.prompt_ms,
                    )

                    # Stage 4: Answer Generation
                    current_stage = "llm"
                    telemetry.llm.model = self._answer_generator.model_name
                    with ExecutionTimer() as timer:
                        answer = self._answer_generator.generate(prompt)
                    latencies.generation_ms = timer.elapsed_ms
                    telemetry.llm.latency_ms = timer.elapsed_ms
                    telemetry.llm.status = "success"

                    # Populate extracted metrics from AnswerGenerator
                    telemetry.llm.ttft_ms = self._answer_generator.last_ttft_ms
                    tokens = self._answer_generator.last_token_usage
                    telemetry.llm.prompt_tokens = tokens.prompt_tokens
                    telemetry.llm.generation_tokens = tokens.generation_tokens
                    telemetry.llm.total_tokens = tokens.total_tokens

                    self._logger.info(
                        "Answer generation completed (latency_ms=%.2f)",
                        latencies.generation_ms,
                    )

                    return answer

            except Exception as exc:
                telemetry.error.occurred = True
                telemetry.error.stage = current_stage
                telemetry.error.message = f"{type(exc).__name__}: {str(exc)}"

                if current_stage == "retrieval":
                    telemetry.retrieval.status = "failed"
                elif current_stage == "llm":
                    telemetry.llm.status = "failed"

                if not isinstance(exc, ValueError):
                    self._logger.error(
                        "Unhandled failure during pipeline execution: %s",
                        type(exc).__name__,
                        exc_info=True,
                    )
                raise

            finally:
                latencies.total_ms = total_timer.elapsed_ms
                telemetry.total_latency_ms = total_timer.elapsed_ms
                self._last_latencies = latencies
                self._last_telemetry = telemetry
