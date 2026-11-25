# Module: test_observability_runner
# Path: tests/unit/test_observability_runner.py
# Purpose: Unit tests for the observability baseline evaluation runner.
# Description: Validates JSONL evaluation loading, failure resilience,
# progress reporting, and aggregated report generation in isolation.

from __future__ import annotations

from pathlib import Path

import pytest

from rag_performance.metrics_collector import MetricsCollector
from rag_performance.metrics_store import InMemoryMetricsStore
from rag_performance.observability_runner import (
    EvaluationCase,
    ObservabilityBaselineRunner,
)
from rag_performance.telemetry import (
    ErrorMetrics,
    LLMMetrics,
    RequestTelemetry,
    RetrievalMetrics,
)


class FakeRunnerPipeline:
    """Lightweight pipeline double simulating success and failure cases."""

    def __init__(self) -> None:
        self._last_telemetry: RequestTelemetry | None = None

    @property
    def last_telemetry(self) -> RequestTelemetry | None:
        """Return latest recorded telemetry."""
        return self._last_telemetry

    def query(self, question: str, request_id: str | None = None) -> str:
        """Simulate query execution and fail on FAIL request IDs."""
        req_id = request_id or "auto"
        if req_id == "EVAL-FAIL":
            self._last_telemetry = RequestTelemetry(
                request_id=req_id,
                total_latency_ms=10.0,
                error=ErrorMetrics(
                    occurred=True,
                    stage="llm",
                    message="RuntimeError: Simulated failure",
                ),
            )
            raise RuntimeError("Simulated failure")

        self._last_telemetry = RequestTelemetry(
            request_id=req_id,
            total_latency_ms=50.0,
            retrieval=RetrievalMetrics(
                latency_ms=5.0,
                k_configured=4,
                documents_retrieved=4,
                status="success",
            ),
            llm=LLMMetrics(
                model="llama3.2:3b",
                latency_ms=45.0,
                prompt_tokens=100,
                generation_tokens=25,
                total_tokens=125,
                status="success",
            ),
        )
        return f"Answer: {question}"


def test_load_evaluation_cases_parses_jsonl(tmp_path: Path) -> None:
    """Load evaluation cases from JSONL while skipping blank lines."""
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '{"id":"EVAL-001","question":"Q1?","answerable":true}\n\n'
        '{"id":"EVAL-012","question":"Q2?","answerable":false}\n',
        encoding="utf-8",
    )

    cases = ObservabilityBaselineRunner.load_evaluation_cases(dataset)

    assert len(cases) == 2
    assert cases[0] == EvaluationCase(
        eval_id="EVAL-001", question="Q1?", answerable=True
    )
    assert cases[1] == EvaluationCase(
        eval_id="EVAL-012", question="Q2?", answerable=False
    )


def test_load_evaluation_cases_raises_when_missing(tmp_path: Path) -> None:
    """Raise FileNotFoundError when dataset file does not exist."""
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="Evaluation dataset not found"):
        ObservabilityBaselineRunner.load_evaluation_cases(missing)


def test_run_executes_cases_handles_failure_and_aggregates() -> None:
    """Continue past individual failures and aggregate all results."""
    collector = MetricsCollector(store=InMemoryMetricsStore())
    pipeline = FakeRunnerPipeline()
    runner = ObservabilityBaselineRunner(
        pipeline=pipeline,
        collector=collector,
    )

    cases = [
        EvaluationCase(eval_id="EVAL-001", question="Q1", answerable=True),
        EvaluationCase(eval_id="EVAL-FAIL", question="Q2", answerable=True),
        EvaluationCase(eval_id="EVAL-002", question="Q3", answerable=True),
    ]
    progress_logs: list[str] = []

    report = runner.run(
        cases=cases,
        reset_store=True,
        progress_callback=progress_logs.append,
    )

    assert report.total_requests == 3
    assert report.successful_requests == 2
    assert report.failed_requests == 1
    assert report.total_latency_ms.mean == pytest.approx(50.0)
    assert report.total_tokens.mean == pytest.approx(125.0)
    assert len(progress_logs) == 6
    assert collector.get_by_request_id("EVAL-001") is not None
    assert collector.get_by_request_id("EVAL-FAIL") is not None
