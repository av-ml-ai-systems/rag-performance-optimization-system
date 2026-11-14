# Module: test_logging
# Path: tests/unit/test_logging.py
# Purpose: Unit tests for request identification and contextual logging.
# Description: Validates RequestContext state isolation, scoping, automatic cleanup,
# and RequestIdFilter injection into LogRecord objects.

from __future__ import annotations

import logging

import pytest

from rag_performance.logging import (
    RequestContext,
    RequestIdFilter,
    get_logger,
)


def test_request_context_explicit_id() -> None:
    context = RequestContext()
    assigned = context.set_request_id("test-request-123")

    assert assigned == "test-request-123"
    assert context.get_request_id() == "test-request-123"

    context.clear()
    assert context.get_request_id() is None


def test_request_context_generated_id() -> None:
    context = RequestContext()
    generated_none = context.set_request_id(None)

    assert isinstance(generated_none, str)
    assert len(generated_none) == 32
    assert context.get_request_id() == generated_none

    context.clear()
    generated_blank = context.set_request_id("   ")
    assert isinstance(generated_blank, str)
    assert len(generated_blank) == 32
    assert context.get_request_id() == generated_blank

    context.clear()


def test_request_context_scope_cleanup() -> None:
    context = RequestContext()
    assert context.get_request_id() is None

    with context.scope("scoped-id-456") as active_id:
        assert active_id == "scoped-id-456"
        assert context.get_request_id() == "scoped-id-456"

    assert context.get_request_id() is None


def test_request_context_scope_cleanup_on_exception() -> None:
    context = RequestContext()
    assert context.get_request_id() is None

    with pytest.raises(RuntimeError, match="Pipeline failure"):
        with context.scope("failing-id-789"):
            assert context.get_request_id() == "failing-id-789"
            raise RuntimeError("Pipeline failure")

    assert context.get_request_id() is None


def test_request_id_filter_with_active_context() -> None:
    context = RequestContext()
    context.set_request_id("filter-test-id")

    log_filter = RequestIdFilter(context=context)
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    assert log_filter.filter(record) is True
    assert getattr(record, "request_id", None) == "filter-test-id"

    context.clear()


def test_request_id_filter_without_active_context() -> None:
    context = RequestContext()
    log_filter = RequestIdFilter(context=context)
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    assert log_filter.filter(record) is True
    assert getattr(record, "request_id", None) == "no_request_id"


def test_get_logger_filter_attachment() -> None:
    context = RequestContext()
    logger_name = "test_custom_logger"
    logger = get_logger(logger_name, context=context)

    matching_filters = [f for f in logger.filters if isinstance(f, RequestIdFilter)]
    assert len(matching_filters) == 1

    logger_again = get_logger(logger_name, context=context)
    matching_filters_again = [
        f for f in logger_again.filters if isinstance(f, RequestIdFilter)
    ]
    assert len(matching_filters_again) == 1
