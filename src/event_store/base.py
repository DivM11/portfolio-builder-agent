"""Event store interfaces and no-op implementation."""

from __future__ import annotations

from typing import Protocol

from src.event_store.models import EventRecord


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

    def close(self) -> None:
        return None
