"""Buffered event store wrapper for lower request-path latency."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

from src.event_store.base import EventStore, MonitoringStore
from src.event_store.models import AgentPerformanceRecord, EventRecord, LLMCallRecord, ToolCallRecord


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
        self._llm_buffer: deque[LLMCallRecord] = deque()
        self._tool_buffer: deque[ToolCallRecord] = deque()
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
            pending_llm = list(self._llm_buffer)
            self._llm_buffer.clear()
            pending_tool = list(self._tool_buffer)
            self._tool_buffer.clear()
        for event in pending:
            self._store.record(event)
        if isinstance(self._store, MonitoringStore):
            for llm_record in pending_llm:
                self._store.record_llm_call(llm_record)
            for tool_record in pending_tool:
                self._store.record_tool_call(tool_record)
        if not self._closed:
            self._schedule()

    def query(
        self,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return self._store.query(session_id=session_id, event_type=event_type, since=since, limit=limit)

    # ------------------------------------------------------------------
    # MonitoringStore forwarding — LLM and tool calls are buffered for
    # reduced write latency; agent_performance is low-volume (once per
    # run) so it remains a direct pass-through.
    # ------------------------------------------------------------------

    def record_llm_call(self, record: LLMCallRecord) -> None:
        if not isinstance(self._store, MonitoringStore):
            return
        with self._lock:
            self._llm_buffer.append(record)
            should_flush = len(self._llm_buffer) >= self._max_buffer_size
        if should_flush:
            self.flush()

    def record_tool_call(self, record: ToolCallRecord) -> None:
        if not isinstance(self._store, MonitoringStore):
            return
        with self._lock:
            self._tool_buffer.append(record)
            should_flush = len(self._tool_buffer) >= self._max_buffer_size
        if should_flush:
            self.flush()

    def record_agent_performance(self, record: AgentPerformanceRecord) -> None:
        if isinstance(self._store, MonitoringStore):
            self._store.record_agent_performance(record)

    def query_llm_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        **_kwargs: Any,
    ) -> list[LLMCallRecord]:
        if isinstance(self._store, MonitoringStore):
            return self._store.query_llm_calls(session_id=session_id, run_id=run_id, limit=limit)
        return []

    def query_tool_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        **_kwargs: Any,
    ) -> list[ToolCallRecord]:
        if isinstance(self._store, MonitoringStore):
            return self._store.query_tool_calls(session_id=session_id, run_id=run_id, limit=limit)
        return []

    def query_agent_performance(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        **_kwargs: Any,
    ) -> list[AgentPerformanceRecord]:
        if isinstance(self._store, MonitoringStore):
            return self._store.query_agent_performance(session_id=session_id, run_id=run_id, limit=limit)
        return []

    def close(self) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        self.flush()
        self._store.close()
