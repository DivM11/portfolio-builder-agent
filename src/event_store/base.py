"""Event store interfaces and no-op implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.event_store.models import (
    AgentPerformanceRecord,
    EventRecord,
    LLMCallRecord,
    ToolCallRecord,
)


@runtime_checkable
class EventStore(Protocol):
    def record(self, event: EventRecord) -> None:
        raise NotImplementedError

    def query(
        self,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@runtime_checkable
class MonitoringStore(EventStore, Protocol):
    """Extended store that records purpose-built monitoring tables."""

    def record_llm_call(self, record: LLMCallRecord) -> None:
        raise NotImplementedError

    def record_tool_call(self, record: ToolCallRecord) -> None:
        raise NotImplementedError

    def record_agent_performance(self, record: AgentPerformanceRecord) -> None:
        raise NotImplementedError

    def query_llm_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[LLMCallRecord]:
        raise NotImplementedError

    def query_tool_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolCallRecord]:
        raise NotImplementedError

    def query_agent_performance(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentPerformanceRecord]:
        raise NotImplementedError


class NullEventStore:
    def record(self, event: EventRecord) -> None:
        return None

    def query(
        self,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return []

    def record_llm_call(self, record: LLMCallRecord) -> None:
        return None

    def record_tool_call(self, record: ToolCallRecord) -> None:
        return None

    def record_agent_performance(self, record: AgentPerformanceRecord) -> None:
        return None

    def query_llm_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[LLMCallRecord]:
        return []

    def query_tool_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolCallRecord]:
        return []

    def query_agent_performance(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentPerformanceRecord]:
        return []

    def close(self) -> None:
        return None
