# Module: test_metrics_reporting
# Path: tests/unit/test_metrics_reporting.py
# Purpose: Unit tests for telemetry statistical aggregation and reporting.
# Description: Validates descriptive statistics, sample-size thresholds for
# standard deviation and P90/P95 percentiles, None filtering, and formatting.

from __future__ import annotations

import pytest

from rag_performance.metrics_reporting import MetricsAggregator
from rag_performance.telemetry import (
    ContextMetrics,
    ErrorMetrics,
    LLMMetrics,
    PromptMetrics,
    RequestTelemetry,
    RetrievalMetrics,
)


def test_summarize_empty_values() -> None:
    """Return zero count and None statistics when input list is empty."""
    aggregator = MetricsAggregator()
    summary = aggregator.summarize_values([])

    assert summary.count == 0
    assert summary.mean is None
    assert summary.median is None
    assert summary.minimum is None
    assert summary.maximum is None
    assert summary.std_dev is None
    assert summary.p90 is None
    assert summary.p95 is None


def test_summarize_single_value() -> None:
    """Omit std_dev and percentiles when sample size is 1."""
    aggregator = MetricsAggregator()
    summary = aggregator.summarize_values([42.0])

    assert summary.count == 1
    assert summary.mean == 42.0
    assert summary.median == 42.0
    assert summary.minimum == 42.0
    assert summary.maximum == 42.0
    assert summary.std_dev is None
    assert summary.p90 is None
    assert summary.p95 is None


def test_summarize_small_sample_omits_percentiles() -> None:
    """Compute std_dev for n=5 while keeping P90 and P95 as None."""
    aggregator = MetricsAggregator()
    summary = aggregator.summarize_values([10.0, 20.0, 30.0, 40.0, 50.0])

    assert summary.count == 5
    assert summary.mean == pytest.approx(30.0)
    assert summary.median == pytest.approx(30.0)
    assert summary.minimum == 10.0
    assert summary.maximum == 50.0
    assert summary.std_dev is not None
    assert summary.std_dev > 0.0
    assert summary.p90 is None
    assert summary.p95 is None


def test_summarize_medium_sample_computes_p90_only() -> None:
    """Compute P90 at n=10 while keeping P95 as None until n>=20."""
    aggregator = MetricsAggregator()
    values = [float(i) for i in range(1, 11)]
    summary = aggregator.summarize_values(values)

    assert summary.count == 10
    assert summary.std_dev is not None
    assert summary.p90 == pytest.approx(9.1)
    assert summary.p95 is None


def test_summarize_large_sample_computes_all_stats() -> None:
    """Compute all statistics including P90 and P95 when n=20."""
    aggregator = MetricsAggregator()
    values = [float(i) for i in range(1, 21)]
    summary = aggregator.summarize_values(values)

    assert summary.count == 20
    assert summary.mean == pytest.approx(10.5)
    assert summary.median == pytest.approx(10.5)
    assert summary.minimum == 1.0
    assert summary.maximum == 20.0
    assert summary.std_dev is not None
    assert summary.p90 == pytest.approx(18.1)
    assert summary.p95 == pytest.approx(19.05)


def test_aggregate_handles_mixed_records_and_formats_report() -> None:
    """Separate failed runs, skip None metrics, and render report output."""
    rec_with_tokens = RequestTelemetry(
        request_id="req-1",
        total_latency_ms=100.0,
        retrieval=RetrievalMetrics(latency_ms=10.0, status="success"),
        context=ContextMetrics(latency_ms=5.0, context_size_chars=500),
        prompt=PromptMetrics(latency_ms=2.0),
        llm=LLMMetrics(
            model="llama3.2:3b",
            latency_ms=83.0,
            ttft_ms=25.0,
            prompt_tokens=100,
            generation_tokens=50,
            total_tokens=150,
            status="success",
        ),
    ).to_dict()

    rec_missing_optional = RequestTelemetry(
        request_id="req-2",
        total_latency_ms=200.0,
        retrieval=RetrievalMetrics(latency_ms=20.0, status="success"),
        context=ContextMetrics(latency_ms=5.0, context_size_chars=700),
        prompt=PromptMetrics(latency_ms=4.0),
        llm=LLMMetrics(
            model="llama3.2:3b",
            latency_ms=171.0,
            ttft_ms=None,
            prompt_tokens=None,
            generation_tokens=None,
            total_tokens=None,
            status="success",
        ),
    ).to_dict()

    rec_failed = RequestTelemetry(
        request_id="req-3",
        total_latency_ms=50.0,
        error=ErrorMetrics(occurred=True, stage="llm", message="Timeout"),
    ).to_dict()

    aggregator = MetricsAggregator()
    report = aggregator.aggregate([rec_with_tokens, rec_missing_optional, rec_failed])

    assert report.total_requests == 3
    assert report.successful_requests == 2
    assert report.failed_requests == 1

    assert report.total_latency_ms.count == 2
    assert report.total_latency_ms.mean == pytest.approx(150.0)
    assert report.retrieval_latency_ms.mean == pytest.approx(15.0)
    assert report.context_size_chars.mean == pytest.approx(600.0)

    # Unavailable None fields must be skipped rather than counted as zero
    assert report.ttft_ms.count == 1
    assert report.ttft_ms.mean == pytest.approx(25.0)
    assert report.total_tokens.count == 1
    assert report.total_tokens.mean == pytest.approx(150.0)

    report_dict = report.to_dict()
    assert report_dict["requests"]["successful"] == 2
    assert report_dict["requests"]["failed"] == 1

    text_report = report.format_text()
    assert "=== RAG Performance Summary ===" in text_report
    assert "Total Latency (ms) (n=2):" in text_report
    assert "TTFT (ms) (n=1):" in text_report
