# Module: metrics_reporting
# Path: src/rag_performance/metrics_reporting.py
# Purpose: Statistical aggregation and performance reporting for RAG telemetry.
# Description: Computes descriptive statistics (mean, median, min, max, stdev,
# P90, P95) over collected telemetry records with sample-size safeguards.

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

MIN_SAMPLES_STD_DEV = 2
MIN_SAMPLES_P90 = 10
MIN_SAMPLES_P95 = 20


@dataclass
class MetricSummary:
    """Descriptive statistics summary for a numeric metric series."""

    count: int = 0
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    std_dev: float | None = None
    p90: float | None = None
    p95: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary statistics into a dictionary."""
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "std_dev": self.std_dev,
            "p90": self.p90,
            "p95": self.p95,
        }


@dataclass
class PerformanceReport:
    """Consolidated performance summary across multiple RAG requests."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_latency_ms: MetricSummary
    retrieval_latency_ms: MetricSummary
    context_latency_ms: MetricSummary
    prompt_latency_ms: MetricSummary
    generation_latency_ms: MetricSummary
    ttft_ms: MetricSummary
    prompt_tokens: MetricSummary
    generation_tokens: MetricSummary
    total_tokens: MetricSummary
    context_size_chars: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete performance report to a dictionary."""
        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
            },
            "latencies_ms": {
                "total": self.total_latency_ms.to_dict(),
                "retrieval": self.retrieval_latency_ms.to_dict(),
                "context": self.context_latency_ms.to_dict(),
                "prompt": self.prompt_latency_ms.to_dict(),
                "generation": self.generation_latency_ms.to_dict(),
                "ttft": self.ttft_ms.to_dict(),
            },
            "tokens": {
                "prompt_tokens": self.prompt_tokens.to_dict(),
                "generation_tokens": self.generation_tokens.to_dict(),
                "total_tokens": self.total_tokens.to_dict(),
            },
            "context": {
                "context_size_chars": self.context_size_chars.to_dict(),
            },
        }

    def format_text(self) -> str:
        """Render a concise human-readable text performance report."""
        lines = [
            "=== RAG Performance Summary ===",
            (
                f"Requests: total={self.total_requests}, "
                f"successful={self.successful_requests}, "
                f"failed={self.failed_requests}"
            ),
            self._format_metric_line("Total Latency (ms)", self.total_latency_ms),
            self._format_metric_line(
                "Retrieval Latency (ms)", self.retrieval_latency_ms
            ),
            self._format_metric_line("Context Latency (ms)", self.context_latency_ms),
            self._format_metric_line("Prompt Latency (ms)", self.prompt_latency_ms),
            self._format_metric_line(
                "Generation Latency (ms)", self.generation_latency_ms
            ),
            self._format_metric_line("TTFT (ms)", self.ttft_ms),
            self._format_metric_line("Total Tokens", self.total_tokens),
            self._format_metric_line("Context Chars", self.context_size_chars),
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_metric_line(label: str, summary: MetricSummary) -> str:
        """Format a single MetricSummary row safely handling None values."""
        if summary.count == 0:
            return f"{label}: no data (n=0)"

        def _fmt(val: float | None) -> str:
            return f"{val:.2f}" if val is not None else "N/A"

        return (
            f"{label} (n={summary.count}): "
            f"mean={_fmt(summary.mean)}, "
            f"median={_fmt(summary.median)}, "
            f"min={_fmt(summary.minimum)}, "
            f"max={_fmt(summary.maximum)}, "
            f"stdev={_fmt(summary.std_dev)}, "
            f"p90={_fmt(summary.p90)}, "
            f"p95={_fmt(summary.p95)}"
        )


class MetricsAggregator:
    """Compute descriptive statistics over collected RAG telemetry records."""

    def __init__(
        self,
        min_samples_std_dev: int = MIN_SAMPLES_STD_DEV,
        min_samples_p90: int = MIN_SAMPLES_P90,
        min_samples_p95: int = MIN_SAMPLES_P95,
    ) -> None:
        """Initialize aggregator with sample-size threshold safeguards."""
        self._min_std_dev = min_samples_std_dev
        self._min_p90 = min_samples_p90
        self._min_p95 = min_samples_p95

    def summarize_values(self, values: list[float]) -> MetricSummary:
        """Compute summary statistics for a sequence of numeric values."""
        n = len(values)
        if n == 0:
            return MetricSummary()

        sorted_vals = sorted(values)
        mean_val = statistics.mean(sorted_vals)
        median_val = statistics.median(sorted_vals)
        min_val = sorted_vals[0]
        max_val = sorted_vals[-1]

        std_dev_val = statistics.stdev(sorted_vals) if n >= self._min_std_dev else None
        p90_val = self._percentile(sorted_vals, 90.0) if n >= self._min_p90 else None
        p95_val = self._percentile(sorted_vals, 95.0) if n >= self._min_p95 else None

        return MetricSummary(
            count=n,
            mean=mean_val,
            median=median_val,
            minimum=min_val,
            maximum=max_val,
            std_dev=std_dev_val,
            p90=p90_val,
            p95=p95_val,
        )

    def aggregate(self, records: list[dict[str, Any]]) -> PerformanceReport:
        """Aggregate consolidated request records into a PerformanceReport."""
        total_count = len(records)
        successful = [
            rec
            for rec in records
            if not bool(rec.get("error", {}).get("occurred", False))
        ]
        failed_count = total_count - len(successful)

        total_lat = self._extract_series(successful, ("total_latency_ms",))
        ret_lat = self._extract_series(
            successful, ("retrieval", "retrieval_latency_ms")
        )
        ctx_lat = self._extract_series(
            successful, ("context", "context_construction_latency_ms")
        )
        pmt_lat = self._extract_series(
            successful, ("prompt", "prompt_construction_latency_ms")
        )
        gen_lat = self._extract_series(successful, ("llm", "generation_latency_ms"))
        ttft_vals = self._extract_series(successful, ("llm", "ttft_ms"))

        prompt_tok = self._extract_series(successful, ("llm", "prompt_tokens"))
        gen_tok = self._extract_series(successful, ("llm", "generation_tokens"))
        total_tok = self._extract_series(successful, ("llm", "total_tokens"))
        ctx_chars = self._extract_series(successful, ("context", "context_size_chars"))

        return PerformanceReport(
            total_requests=total_count,
            successful_requests=len(successful),
            failed_requests=failed_count,
            total_latency_ms=self.summarize_values(total_lat),
            retrieval_latency_ms=self.summarize_values(ret_lat),
            context_latency_ms=self.summarize_values(ctx_lat),
            prompt_latency_ms=self.summarize_values(pmt_lat),
            generation_latency_ms=self.summarize_values(gen_lat),
            ttft_ms=self.summarize_values(ttft_vals),
            prompt_tokens=self.summarize_values(prompt_tok),
            generation_tokens=self.summarize_values(gen_tok),
            total_tokens=self.summarize_values(total_tok),
            context_size_chars=self.summarize_values(ctx_chars),
        )

    @staticmethod
    def _extract_series(
        records: list[dict[str, Any]],
        path: tuple[str, ...],
    ) -> list[float]:
        """Extract numeric values along a nested key path, skipping None."""
        extracted: list[float] = []
        for rec in records:
            current: Any = rec
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if isinstance(current, (int, float)) and not isinstance(current, bool):
                extracted.append(float(current))
        return extracted

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        """Compute percentile via linear interpolation on sorted values."""
        k = (len(sorted_values) - 1) * (percentile / 100.0)
        lower_idx = math.floor(k)
        upper_idx = math.ceil(k)
        if lower_idx == upper_idx:
            return sorted_values[int(k)]
        fraction = k - lower_idx
        lower_val = sorted_values[lower_idx]
        upper_val = sorted_values[upper_idx]
        return lower_val + fraction * (upper_val - lower_val)
