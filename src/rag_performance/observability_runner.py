# Module: observability_runner
# Path: src/rag_performance/observability_runner.py
# Purpose: Execute evaluation queries to collect an observability dataset.
# Description: Loads evaluation questions, executes the RAG pipeline via
# MetricsCollector against the persisted ChromaDB, and generates a report.

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_performance.answer_generator import AnswerGenerator
from rag_performance.context_builder import ContextBuilder
from rag_performance.embeddings import OllamaEmbeddingProvider
from rag_performance.logging import get_logger
from rag_performance.metrics_collector import (
    MetricsCollector,
    ObservablePipeline,
)
from rag_performance.metrics_reporting import (
    MetricsAggregator,
    PerformanceReport,
)
from rag_performance.metrics_store import JsonlMetricsStore
from rag_performance.prompt_builder import PromptBuilder
from rag_performance.rag_pipeline import RAGPipeline
from rag_performance.retriever import Retriever
from rag_performance.vector_store import VectorStore

logger = get_logger(__name__)

DEFAULT_QUESTIONS_PATH = Path("data/evaluation/questions.jsonl")
DEFAULT_BASELINE_METRICS_PATH = Path("data/telemetry/baseline_metrics.jsonl")
DEFAULT_BASELINE_REPORT_PATH = Path("data/telemetry/baseline_report.json")


@dataclass(frozen=True)
class EvaluationCase:
    """Represent a single evaluation query from the benchmark dataset."""

    eval_id: str
    question: str
    answerable: bool


class ObservabilityBaselineRunner:
    """Coordinate execution of evaluation queries to collect telemetry."""

    def __init__(
        self,
        pipeline: ObservablePipeline,
        collector: MetricsCollector,
        aggregator: MetricsAggregator | None = None,
    ) -> None:
        """Initialize the runner with pipeline, collector, and aggregator."""
        self._pipeline = pipeline
        self._collector = collector
        self._aggregator = aggregator or MetricsAggregator()

    @staticmethod
    def load_evaluation_cases(
        questions_path: Path = DEFAULT_QUESTIONS_PATH,
    ) -> list[EvaluationCase]:
        """Load evaluation cases from a JSONL dataset file."""
        if not questions_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {questions_path}")

        cases: list[EvaluationCase] = []
        with questions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                item: Any = json.loads(stripped)
                if not isinstance(item, dict):
                    continue
                cases.append(
                    EvaluationCase(
                        eval_id=str(item["id"]),
                        question=str(item["question"]),
                        answerable=bool(item.get("answerable", True)),
                    )
                )
        return cases

    def run(
        self,
        cases: list[EvaluationCase],
        reset_store: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> PerformanceReport:
        """Execute evaluation cases and return an aggregated report."""
        if reset_store:
            self._collector.clear()

        total = len(cases)
        for index, case in enumerate(cases, start=1):
            if progress_callback is not None:
                progress_callback(f"[{index}/{total}] Running {case.eval_id}...")
            try:
                self._collector.run_and_collect(
                    pipeline=self._pipeline,
                    question=case.question,
                    request_id=case.eval_id,
                )
            except Exception as exc:
                logger.warning(
                    "Evaluation query failed (eval_id=%s, error=%s)",
                    case.eval_id,
                    type(exc).__name__,
                )

            telemetry = self._pipeline.last_telemetry
            if progress_callback is not None and telemetry is not None:
                status = "FAILED" if telemetry.error.occurred else "OK"
                progress_callback(
                    f"[{index}/{total}] {case.eval_id} {status} "
                    f"({telemetry.total_latency_ms:.1f} ms)"
                )

        records = self._collector.get_all()
        return self._aggregator.aggregate(records)


def run_baseline_cli() -> None:
    """Execute the 12-question baseline observability collection via CLI."""
    embedding_provider = OllamaEmbeddingProvider(model="nomic-embed-text:latest")
    vector_store = VectorStore(
        embedding_provider=embedding_provider,
        collection_name="rag_knowledge_base",
        persist_directory="data/chroma_db",
    )

    doc_count = vector_store.count_documents()
    if doc_count == 0:
        raise RuntimeError(
            "ChromaDB collection 'rag_knowledge_base' in data/chroma_db "
            "is empty. Ingest documents before running baseline collection."
        )

    print(f"Connected to ChromaDB (records={doc_count}).")

    pipeline = RAGPipeline(
        retriever=Retriever(vector_store=vector_store, k=4),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        answer_generator=AnswerGenerator(model="llama3.2:3b"),
    )

    store = JsonlMetricsStore(DEFAULT_BASELINE_METRICS_PATH)
    collector = MetricsCollector(store=store)
    runner = ObservabilityBaselineRunner(
        pipeline=pipeline,
        collector=collector,
    )

    cases = runner.load_evaluation_cases(DEFAULT_QUESTIONS_PATH)
    print(f"Loaded {len(cases)} evaluation queries. Starting execution...")

    report = runner.run(cases=cases, reset_store=True, progress_callback=print)

    DEFAULT_BASELINE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_BASELINE_REPORT_PATH.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )

    print("\n" + report.format_text())
    print(f"\nSaved telemetry dataset to: {DEFAULT_BASELINE_METRICS_PATH}")
    print(f"Saved baseline report to: {DEFAULT_BASELINE_REPORT_PATH}")


if __name__ == "__main__":
    run_baseline_cli()
