"""Tests for LLMService event instrumentation."""

from src.event_store.base import EventStore
from src.llm_service import LLMService


class DummyRawResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def parse(self):
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 10}}


class DummyWithRaw:
    def create(self, **_kwargs):
        return DummyRawResponse()


class DummyCompletions:
    def __init__(self):
        self.with_raw_response = DummyWithRaw()


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyClient:
    def __init__(self):
        self.chat = DummyChat()


class RecordingStore(EventStore):
    def __init__(self) -> None:
        self.events = []

    def record(self, event):
        self.events.append(event)

    def query(self, **_kwargs):
        return list(self.events)

    def close(self):
        return None


def test_llm_service_records_request_and_response_events() -> None:
    store = RecordingStore()
    service = LLMService(DummyClient(), event_store=store)

    service.complete(
        request_name="ticker_generation",
        model="anthropic/claude-3.5-haiku",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": "hi"}],
        session_id="session-1",
        run_id="run-1",
    )

    assert len(store.events) == 2
    assert store.events[0].event_type == "llm_request"
    assert store.events[1].event_type == "llm_response"
    assert store.events[1].status_code == 200
    assert store.events[1].raw_output == "ok"
