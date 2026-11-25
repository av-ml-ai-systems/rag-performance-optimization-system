# Module: test_metrics_store
# Path: tests/unit/test_metrics_store.py
# Purpose: Unit tests for local JSONL telemetry persistence.
# Description: Validates JsonlMetricsStore save, append, defensive load,
# directory creation, and clear behavior using temporary test paths.

from __future__ import annotations

from pathlib import Path

from rag_performance.metrics_store import JsonlMetricsStore
from rag_performance.telemetry import (
    ContextMetrics,
    LLMMetrics,
    RequestTelemetry,
    RetrievalMetrics,
)


def test_load_all_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """Return an empty list when the telemetry JSONL file does not exist."""
    store = JsonlMetricsStore(tmp_path / "missing" / "metrics.jsonl")

    assert store.load_all() == []


def test_save_creates_parent_dirs_and_persists_record(tmp_path: Path) -> None:
    """Create parent directories and persist RequestTelemetry as JSONL."""
    target_path = tmp_path / "nested" / "telemetry" / "metrics.jsonl"
    store = JsonlMetricsStore(target_path)

    telemetry = RequestTelemetry(
        request_id="req-save-001",
        total_latency_ms=150.25,
        retrieval=RetrievalMetrics(
            latency_ms=12.5,
            k_configured=4,
            documents_retrieved=3,
            status="success",
        ),
        context=ContextMetrics(
            latency_ms=2.0,
            chunks_used=3,
            context_size_chars=400,
            context_size_tokens_estimated=100,
        ),
        llm=LLMMetrics(
            model="llama3.2:3b",
            latency_ms=130.0,
            prompt_tokens=120,
            generation_tokens=30,
            total_tokens=150,
            status="success",
        ),
    )

    store.save(telemetry)

    assert target_path.exists()
    records = store.load_all()
    assert len(records) == 1
    assert records[0]["request_id"] == "req-save-001"
    assert records[0]["total_latency_ms"] == 150.25
    assert records[0]["retrieval"]["documents_retrieved"] == 3
    assert records[0]["llm"]["total_tokens"] == 150


def test_save_appends_multiple_records_in_order(tmp_path: Path) -> None:
    """Append multiple records sequentially without overwriting prior lines."""
    store = JsonlMetricsStore(tmp_path / "metrics.jsonl")

    store.save(RequestTelemetry(request_id="req-1", total_latency_ms=10.0))
    store.save(RequestTelemetry(request_id="req-2", total_latency_ms=20.0))

    records = store.load_all()
    assert len(records) == 2
    assert records[0]["request_id"] == "req-1"
    assert records[1]["request_id"] == "req-2"


def test_load_all_skips_blank_lines_and_non_dict_entries(
    tmp_path: Path,
) -> None:
    """Ignore empty lines and non-object JSON lines when loading records."""
    target_path = tmp_path / "metrics.jsonl"
    target_path.write_text(
        '\n[1, 2, 3]\n{"request_id": "req-valid", "total_latency_ms": 5.0}\n\n',
        encoding="utf-8",
    )
    store = JsonlMetricsStore(target_path)

    records = store.load_all()
    assert len(records) == 1
    assert records[0]["request_id"] == "req-valid"


def test_clear_removes_persisted_file(tmp_path: Path) -> None:
    """Delete telemetry file on clear and remain safe if already missing."""
    store = JsonlMetricsStore(tmp_path / "metrics.jsonl")
    store.save(RequestTelemetry(request_id="req-clear"))
    assert store.file_path.exists()

    store.clear()
    assert not store.file_path.exists()
    assert store.load_all() == []

    # Calling clear again on a missing file should not raise
    store.clear()
