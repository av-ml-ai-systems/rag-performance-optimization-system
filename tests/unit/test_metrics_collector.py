# Module: test_metrics_collector
# Path: tests/unit/test_metrics_collector.py
# Purpose: Unit tests for telemetry collection and multi-request retrieval.
# Description: Validates MetricsCollector execution wrapping, failure-resilient
# collection, filtering across requests, and MetricsRepository compliance.

from __future__ import annotations

from pathlib import Path

import pytest

from rag_performance.metrics_collector import MetricsCollector
from rag_performance.metrics_store import (
    InMemoryMetricsStore,
    JsonlMetricsStore,
    MetricsRepository,
)
from rag_performance.telemetry import ErrorMetrics, RequestTelemetry


class FakePipeline:
    """Lightweight test double satisfying the ObservablePipeline protocol."""

    def __init__(self, should_fail: bool = False) -> None:
        self._should_fail = should_fail
        self._last_telemetry: RequestTelemetry | None = None

    @property
    def last_telemetry(self) -> RequestTelemetry | None:
        """Return the latest recorded telemetry."""
        return self._last_telemetry

    def query(self, question: str, request_id: str | None = None) -> str:
        """Simulate a successful or failing RAG query execution."""
        req_id = request_id or "auto-generated-id"
        if self._should_fail:
            self._last_telemetry = RequestTelemetry(
                request_id=req_id,
                total_latency_ms=15.0,
                error=ErrorMetrics(
                    occurred=True,
                    stage="llm",
                    message="RuntimeError: Model offline",
                ),
            )
            raise RuntimeError("Model offline")

        self._last_telemetry = RequestTelemetry(
            request_id=req_id,
            total_latency_ms=42.5,
        )
        return f"Answer to: {question}"


def test_stores_satisfy_repository_protocol(tmp_path: Path) -> None:
    """Verify both store implementations comply with MetricsRepository."""
    in_memory = InMemoryMetricsStore()
    jsonl = JsonlMetricsStore(tmp_path / "metrics.jsonl")

    assert isinstance(in_memory, MetricsRepository)
    assert isinstance(jsonl, MetricsRepository)


def test_collect_and_retrieve_multiple_requests() -> None:
    """Verify collection and filtering across successful and failed runs."""
    store = InMemoryMetricsStore()
    collector = MetricsCollector(store=store)

    collector.collect(RequestTelemetry(request_id="req-ok-1", total_latency_ms=10.0))
    collector.collect(
        RequestTelemetry(
            request_id="req-err-1",
            total_latency_ms=5.0,
            error=ErrorMetrics(occurred=True, stage="retrieval", message="Err"),
        )
    )
    collector.collect(RequestTelemetry(request_id="req-ok-2", total_latency_ms=20.0))

    assert len(collector.get_all()) == 3
    assert len(collector.get_successful()) == 2
    assert len(collector.get_failed()) == 1

    match = collector.get_by_request_id("req-err-1")
    assert match is not None
    assert match["error"]["stage"] == "retrieval"

    collector.clear()
    assert collector.get_all() == []


def test_run_and_collect_persists_telemetry_on_success() -> None:
    """Verify run_and_collect returns answer and persists telemetry."""
    collector = MetricsCollector(store=InMemoryMetricsStore())
    pipeline = FakePipeline(should_fail=False)

    answer = collector.run_and_collect(
        pipeline=pipeline,
        question="What is RAG?",
        request_id="run-success-001",
    )

    assert answer == "Answer to: What is RAG?"
    records = collector.get_successful()
    assert len(records) == 1
    assert records[0]["request_id"] == "run-success-001"
    assert records[0]["total_latency_ms"] == 42.5


def test_run_and_collect_persists_partial_telemetry_on_failure() -> None:
    """Verify partial telemetry is saved even when pipeline raises an error."""
    collector = MetricsCollector(store=InMemoryMetricsStore())
    pipeline = FakePipeline(should_fail=True)

    with pytest.raises(RuntimeError, match="Model offline"):
        collector.run_and_collect(
            pipeline=pipeline,
            question="Failing query",
            request_id="run-fail-001",
        )

    failed_records = collector.get_failed()
    assert len(failed_records) == 1
    assert failed_records[0]["request_id"] == "run-fail-001"
    assert failed_records[0]["error"]["occurred"] is True
    assert failed_records[0]["error"]["stage"] == "llm"


def test_get_by_request_id_returns_none_when_missing() -> None:
    """Return None when requested correlation ID does not exist."""
    collector = MetricsCollector(store=InMemoryMetricsStore())
    collector.collect(RequestTelemetry(request_id="existing-id"))

    assert collector.get_by_request_id("non-existent-id") is None
