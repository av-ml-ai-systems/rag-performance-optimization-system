# Module: metrics_store
# Path: src/rag_performance/metrics_store.py
# Purpose: Lightweight local persistence for request-level RAG telemetry.
# Description: Defines the MetricsRepository protocol along with
# JsonlMetricsStore and InMemoryMetricsStore implementations.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rag_performance.logging import get_logger
from rag_performance.telemetry import RequestTelemetry

logger = get_logger(__name__)

DEFAULT_METRICS_PATH = Path("data/telemetry/metrics.jsonl")


@runtime_checkable
class MetricsRepository(Protocol):
    """Contract for storing and retrieving request-level telemetry records."""

    def save(self, telemetry: RequestTelemetry) -> None:
        """Persist a single RequestTelemetry record."""
        ...

    def load_all(self) -> list[dict[str, Any]]:
        """Retrieve all persisted telemetry records."""
        ...

    def clear(self) -> None:
        """Clear all persisted telemetry records."""
        ...


class InMemoryMetricsStore:
    """In-memory telemetry store for isolated testing and transient runs."""

    def __init__(self) -> None:
        """Initialize an empty in-memory record list."""
        self._records: list[dict[str, Any]] = []

    def save(self, telemetry: RequestTelemetry) -> None:
        """Append a serialized RequestTelemetry record in memory."""
        self._records.append(telemetry.to_dict())

    def load_all(self) -> list[dict[str, Any]]:
        """Return a shallow copy of stored telemetry records."""
        return list(self._records)

    def clear(self) -> None:
        """Remove all stored records from memory."""
        self._records.clear()


class JsonlMetricsStore:
    """Persist and retrieve consolidated request telemetry in JSONL format."""

    def __init__(self, file_path: Path | str = DEFAULT_METRICS_PATH) -> None:
        """Initialize the store with the target JSONL file path."""
        self._file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        """Return the configured path to the telemetry JSONL file."""
        return self._file_path

    def save(self, telemetry: RequestTelemetry) -> None:
        """Append a consolidated RequestTelemetry record to the JSONL file."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(telemetry.to_dict(), ensure_ascii=False)

        with self._file_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{payload}\n")

        logger.debug(
            "Persisted telemetry record (request_id=%s, path=%s)",
            telemetry.request_id,
            self._file_path,
        )

    def load_all(self) -> list[dict[str, Any]]:
        """Load all persisted telemetry records from the JSONL file."""
        if not self._file_path.exists():
            logger.debug(
                "Telemetry file does not exist yet (path=%s)",
                self._file_path,
            )
            return []

        records: list[dict[str, Any]] = []
        with self._file_path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parsed: Any = json.loads(stripped)
                if not isinstance(parsed, dict):
                    logger.warning(
                        "Skipped non-object JSON record at line %d",
                        line_num,
                    )
                    continue
                records.append(parsed)

        logger.debug(
            "Loaded %d telemetry records (path=%s)",
            len(records),
            self._file_path,
        )
        return records

    def clear(self) -> None:
        """Remove the persisted telemetry file if it exists."""
        if self._file_path.exists():
            self._file_path.unlink()
            logger.info("Cleared telemetry store (path=%s)", self._file_path)
