# Module: telemetry
# Path: src/rag_performance/telemetry.py
# Purpose: Performance timing and measurement primitives for RAG observability.
# Description: Provides a monotonic ExecutionTimer, component-level metrics
# models, token usage structures, and consolidated RequestTelemetry.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import TracebackType
from typing import Any


@dataclass
class TokenUsage:
    """Store input, output, and total token counts from LLM generation."""

    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ComponentLatencies:
    """Store monotonic latency measurements for each RAG pipeline stage.

    All durations are recorded in milliseconds.
    """

    retrieval_ms: float = 0.0
    context_ms: float = 0.0
    prompt_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RetrievalMetrics:
    """Retrieval stage latency and document metadata."""

    latency_ms: float = 0.0
    k_configured: int = 0
    documents_retrieved: int = 0
    status: str = "pending"


@dataclass
class ContextMetrics:
    """Context construction stage latency and size metrics."""

    latency_ms: float = 0.0
    chunks_used: int = 0
    context_size_chars: int = 0
    context_size_tokens_estimated: int = 0


@dataclass
class PromptMetrics:
    """Prompt construction stage latency metrics."""

    latency_ms: float = 0.0


@dataclass
class LLMMetrics:
    """LLM generation stage latency and model metadata."""

    model: str = ""
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    total_tokens: int | None = None
    status: str = "pending"


@dataclass
class ErrorMetrics:
    """Operational error status and diagnostic details."""

    occurred: bool = False
    stage: str | None = None
    message: str | None = None


@dataclass
class RequestTelemetry:
    """Consolidated telemetry record for a single RAG request."""

    request_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_latency_ms: float = 0.0
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    context: ContextMetrics = field(default_factory=ContextMetrics)
    prompt: PromptMetrics = field(default_factory=PromptMetrics)
    llm: LLMMetrics = field(default_factory=LLMMetrics)
    error: ErrorMetrics = field(default_factory=ErrorMetrics)

    def to_dict(self) -> dict[str, Any]:
        """Serialize telemetry record into the approved JSON schema."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "retrieval": {
                "retrieval_latency_ms": self.retrieval.latency_ms,
                "k_configured": self.retrieval.k_configured,
                "documents_retrieved": self.retrieval.documents_retrieved,
                "status": self.retrieval.status,
            },
            "context": {
                "context_construction_latency_ms": self.context.latency_ms,
                "chunks_used": self.context.chunks_used,
                "context_size_chars": self.context.context_size_chars,
                "context_size_tokens_estimated": (
                    self.context.context_size_tokens_estimated
                ),
            },
            "prompt": {
                "prompt_construction_latency_ms": self.prompt.latency_ms,
            },
            "llm": {
                "model": self.llm.model,
                "generation_latency_ms": self.llm.latency_ms,
                "ttft_ms": self.llm.ttft_ms,
                "prompt_tokens": self.llm.prompt_tokens,
                "generation_tokens": self.llm.generation_tokens,
                "total_tokens": self.llm.total_tokens,
                "status": self.llm.status,
            },
            "error": {
                "occurred": self.error.occurred,
                "stage": self.error.stage,
                "message": self.error.message,
            },
            "total_latency_ms": self.total_latency_ms,
        }


class ExecutionTimer:
    """High-resolution monotonic timer for measuring execution latency."""

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._elapsed_ms: float = 0.0

    @property
    def is_running(self) -> bool:
        """Return True if the timer is actively running."""
        return self._start_time is not None

    def start(self) -> None:
        """Record the start time using a monotonic clock."""
        self._start_time = time.perf_counter()
        self._elapsed_ms = 0.0

    def stop(self) -> float:
        """Stop the timer and compute elapsed milliseconds."""
        if self._start_time is None:
            raise RuntimeError("Timer was never started.")
        self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._start_time = None
        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Return the elapsed time in milliseconds."""
        return self._elapsed_ms

    def __enter__(self) -> ExecutionTimer:
        """Enter the runtime context and start the monotonic timer."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context and record elapsed milliseconds."""
        if self._start_time is not None:
            self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
            self._start_time = None
