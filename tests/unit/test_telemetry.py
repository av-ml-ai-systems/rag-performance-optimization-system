# Module: test_telemetry
# Path: tests/unit/test_telemetry.py
# Purpose: Unit tests for performance timing and measurement primitives.
# Description: Validates monotonic ExecutionTimer accuracy, context manager protocol,
# manual start/stop lifecycle, and elapsed time capture upon exception.

from __future__ import annotations

import time

import pytest

from rag_performance.telemetry import ExecutionTimer


def test_execution_timer_context_manager() -> None:
    """Verify context manager records positive elapsed milliseconds."""
    with ExecutionTimer() as timer:
        time.sleep(0.005)

    assert timer.elapsed_ms > 0.0


def test_execution_timer_manual_lifecycle() -> None:
    """Verify manual start and stop sequence computes elapsed time."""
    timer = ExecutionTimer()
    timer.start()
    time.sleep(0.005)
    elapsed = timer.stop()

    assert elapsed > 0.0
    assert timer.elapsed_ms == elapsed


def test_execution_timer_stop_without_start_raises() -> None:
    """Verify calling stop without starting the timer raises RuntimeError."""
    timer = ExecutionTimer()
    with pytest.raises(RuntimeError, match="Timer was never started"):
        timer.stop()


def test_execution_timer_preserves_timing_on_exception() -> None:
    """Verify elapsed duration is captured even if an exception occurs."""
    timer = ExecutionTimer()
    with pytest.raises(ValueError, match="Simulated component crash"):
        with timer:
            time.sleep(0.005)
            raise ValueError("Simulated component crash")

    assert timer.elapsed_ms > 0.0
