# Module: logging
# Path: src/rag_performance/logging.py
# Purpose: Request identification and structured logging for RAG observability.
# Description: Provides encapsulated request_id context management using ContextVar,
# a context manager protocol for query scoping, and a contextual logging filter.

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator


class RequestContext:
    """Encapsulates request correlation state using thread-safe context variables."""

    def __init__(self) -> None:
        self._ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

    def set_request_id(self, request_id: str | None = None) -> str:
        """Set active request ID in context. Generates UUIDv4 if none provided."""
        resolved = (
            request_id.strip()
            if request_id and request_id.strip()
            else uuid.uuid4().hex
        )
        self._ctx.set(resolved)
        return resolved

    def get_request_id(self) -> str | None:
        """Retrieve the active request ID from context."""
        return self._ctx.get()

    def clear(self) -> None:
        """Reset the request ID context."""
        self._ctx.set(None)

    @contextmanager
    def scope(self, request_id: str | None = None) -> Generator[str, None, None]:
        """Context manager to guarantee request_id cleanup after execution."""
        token = self.set_request_id(request_id)
        try:
            yield token
        finally:
            self.clear()


# Default singleton instance for application runtime
default_context = RequestContext()


class RequestIdFilter(logging.Filter):
    """Logging filter that injects the active request_id into each LogRecord."""

    def __init__(self, context: RequestContext | None = None) -> None:
        super().__init__()
        self._context = context or default_context

    def set_context(self, context: RequestContext) -> None:
        """Update the context reference for this filter."""
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self._context.get_request_id() or "no_request_id"
        return True


def get_logger(name: str, context: RequestContext | None = None) -> logging.Logger:
    """Obtain a logger instance equipped with an injected RequestIdFilter."""
    logger = logging.getLogger(name)
    target_context = context or default_context

    for existing_filter in logger.filters:
        if isinstance(existing_filter, RequestIdFilter):
            existing_filter.set_context(target_context)
            return logger

    logger.addFilter(RequestIdFilter(target_context))
    return logger
