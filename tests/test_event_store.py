"""Unit tests for event store backends."""

from src.event_store import create_event_store
from src.event_store.base import NullEventStore
from src.event_store.models import EventRecord


def test_create_event_store_disabled_returns_null_store() -> None:
    store = create_event_store({"enabled": False})
    assert isinstance(store, NullEventStore)


def test_sqlite_event_store_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    store = create_event_store(
        {
            "enabled": True,
            "backend": "sqlite",
            "schema_version": 1,
            "sqlite": {"db_path": str(db_path)},
            "buffer": {"enabled": False},
        }
    )
    event = EventRecord(
        event_type="user_action",
        session_id="session-1",
        run_id="run-1",
        action="new_prompt",
        action_payload={"text": "growth"},
    )

    store.record(event)
    events = store.query(session_id="session-1", event_type="user_action", limit=10)

    assert len(events) == 1
    assert events[0].action == "new_prompt"
    assert events[0].action_payload == {"text": "growth"}
    store.close()
