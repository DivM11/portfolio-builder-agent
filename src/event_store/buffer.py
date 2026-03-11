"""Buffered event store wrapper for lower request-path latency."""

from __future__ import annotations

import threading
from collections import deque

from src.event_store.base import EventStore
from src.event_store.models import EventRecord


class BufferedEventStore:
    def __init__(
        self,
        store: EventStore,
        *,
        flush_interval_seconds: float = 5.0,
        max_buffer_size: int = 100,
    ) -> None:
        self._store = store
        self._flush_interval_seconds = max(0.1, float(flush_interval_seconds))
        self._max_buffer_size = max(1, int(max_buffer_size))
        self._buffer: deque[EventRecord] = deque()
        self._lock = threading.Lock()
        self._closed = False
        self._timer: threading.Timer | None = None
        self._schedule()

    def _schedule(self) -> None:
        if self._closed:
            return
        timer = threading.Timer(self._flush_interval_seconds, self.flush)
        timer.daemon = True
        timer.start()
        self._timer = timer

    def record(self, event: EventRecord) -> None:
        with self._lock:
            self._buffer.append(event)
            should_flush = len(self._buffer) >= self._max_buffer_size
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            pending = list(self._buffer)
            self._buffer.clear()
        for event in pending:
            self._store.record(event)
        if not self._closed:
            self._schedule()

    def query(self, **kwargs):
        return self._store.query(**kwargs)

    def close(self) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        self.flush()
        self._store.close()
