# Module: metrics_collector
# Path: src/rag_performance/metrics_collector.py
# Purpose: Collect and retrieve telemetry records across multiple RAG requests.
# Description: Provides MetricsCollector to record pipeline telemetry and
# query historical runs while keeping RAG components decoupled from storage.

from __future__ import annotations

from typing import Any, Protocol

from rag_performance.logging import get_logger
from rag_performance.metrics_store import JsonlMetricsStore, MetricsRepository
from rag_performance.telemetry import RequestTelemetry

logger = get_logger(__name__)


class ObservablePipeline(Protocol):
    """Minimal contract for a RAG pipeline that exposes RequestTelemetry."""

    @property
    def last_telemetry(self) -> RequestTelemetry | None:
        """Return consolidated telemetry from the most recent execution."""
        ...

    def query(self, question: str, request_id: str | None = None) -> str:
        """Execute a query and return the generated answer."""
        ...


class MetricsCollector:
    """Coordinate collection and retrieval of request-level telemetry."""

    def __init__(self, store: MetricsRepository | None = None) -> None:
        """Initialize the collector with an injected MetricsRepository."""
        self._store: MetricsRepository = store or JsonlMetricsStore()

    @property
    def store(self) -> MetricsRepository:
        """Return the underlying telemetry repository."""
        return self._store

    def collect(self, telemetry: RequestTelemetry) -> None:
        """Persist a single RequestTelemetry record."""
        self._store.save(telemetry)
        logger.debug(
            "Collected telemetry record (request_id=%s)",
            telemetry.request_id,
        )

    def run_and_collect(
        self,
        pipeline: ObservablePipeline,
        question: str,
        request_id: str | None = None,
    ) -> str:
        """Execute a pipeline query and persist its resulting telemetry."""
        try:
            return pipeline.query(question, request_id=request_id)
        finally:
            telemetry = pipeline.last_telemetry
            if telemetry is not None:
                self._store.save(telemetry)

    def get_all(self) -> list[dict[str, Any]]:
        """Retrieve all persisted telemetry records."""
        return self._store.load_all()

    def get_successful(self) -> list[dict[str, Any]]:
        """Retrieve telemetry records for queries without errors."""
        return [
            record
            for record in self._store.load_all()
            if not bool(record.get("error", {}).get("occurred", False))
        ]

    def get_failed(self) -> list[dict[str, Any]]:
        """Retrieve telemetry records for queries that recorded an error."""
        return [
            record
            for record in self._store.load_all()
            if bool(record.get("error", {}).get("occurred", False))
        ]

    def get_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        """Retrieve a specific telemetry record by its request_id."""
        for record in self._store.load_all():
            if record.get("request_id") == request_id:
                return record
        return None

    def clear(self) -> None:
        """Clear all records from the underlying store."""
        self._store.clear()
