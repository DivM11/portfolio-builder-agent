"""Unit tests for shared LLM service."""

from src.event_store.base import NullEventStore
from src.event_store.models import AgentPerformanceRecord, LLMCallRecord, ToolCallRecord
from src.llm_service import LLMService, build_prompt, extract_message_text


class DummyRawResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def parse(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class DummyWithRaw:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **_kwargs):
        self.last_kwargs = dict(_kwargs)
        return DummyRawResponse()


class DummyCompletions:
    def __init__(self):
        self.with_raw_response = DummyWithRaw()

    def create(self, **_kwargs):
        return {"choices": [{"message": {"content": "fallback"}}]}


class DummyCompletionsNoRaw:
    def create(self, **_kwargs):
        return {"choices": [{"message": {"content": "fallback-no-raw"}}]}


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyClient:
    def __init__(self):
        self.chat = DummyChat()


class DummyChatNoRaw:
    def __init__(self):
        self.completions = DummyCompletionsNoRaw()


class DummyClientNoRaw:
    def __init__(self):
        self.chat = DummyChatNoRaw()


def test_build_prompt():
    assert build_prompt("Hello {user_input}", "world") == "Hello world"


def test_extract_message_text_dict():
    assert extract_message_text({"choices": [{"message": {"content": "x"}}]}) == "x"


def test_extract_message_text_none_content_returns_empty_string():
    """Reasoning models may return content=None; extract_message_text must return ''."""
    assert extract_message_text({"choices": [{"message": {"content": None}}]}) == ""


def test_llm_service_complete_with_raw_response():
    service = LLMService(DummyClient())

    response, status = service.complete(
        request_name="test",
        model="anthropic/claude-3.5-haiku",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert status == 200
    assert response["choices"][0]["message"]["content"] == "ok"


def test_llm_service_complete_fallback_without_raw_response():
    service = LLMService(DummyClientNoRaw())

    response, status = service.complete(
        request_name="test",
        model="anthropic/claude-3.5-haiku",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert status is None
    assert response["choices"][0]["message"]["content"] == "fallback-no-raw"


def test_model_name_validation():
    assert LLMService.is_model_name_valid("anthropic/claude-3.5-haiku") is True
    assert LLMService.is_model_name_valid("invalid model") is False


def test_complete_with_tools_passes_reasoning_extra_body_when_configured():
    client = DummyClient()
    service = LLMService(client)

    service.complete_with_tools(
        request_name="tooling",
        model="anthropic/claude-3.5-haiku",
        max_tokens=20,
        temperature=0.2,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        reasoning={"effort": "high", "exclude": False},
    )

    kwargs = client.chat.completions.with_raw_response.last_kwargs
    assert kwargs is not None
    assert kwargs.get("extra_body") == {"reasoning": {"effort": "high", "exclude": False}}


def test_complete_with_tools_passes_response_format_when_configured():
    client = DummyClient()
    service = LLMService(client)

    service.complete_with_tools(
        request_name="tooling",
        model="anthropic/claude-3.5-haiku",
        max_tokens=20,
        temperature=0.2,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        response_format={"type": "json_object"},
    )

    kwargs = client.chat.completions.with_raw_response.last_kwargs
    assert kwargs is not None
    assert kwargs.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# MonitoringStore integration — record_llm_call is emitted
# ---------------------------------------------------------------------------

class CaptureMonitoringStore(NullEventStore):
    """Captures both legacy events and new monitoring records."""

    def __init__(self) -> None:
        self.llm_calls: list[LLMCallRecord] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.perf_records: list[AgentPerformanceRecord] = []

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self.llm_calls.append(record)

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)

    def record_agent_performance(self, record: AgentPerformanceRecord) -> None:
        self.perf_records.append(record)

    def query_llm_calls(self, **kwargs) -> list[LLMCallRecord]:
        return list(self.llm_calls)

    def query_tool_calls(self, **kwargs) -> list[ToolCallRecord]:
        return list(self.tool_calls)

    def query_agent_performance(self, **kwargs) -> list[AgentPerformanceRecord]:
        return list(self.perf_records)


def test_complete_emits_llm_call_to_monitoring_store():
    store = CaptureMonitoringStore()
    service = LLMService(DummyClient(), event_store=store, schema_version=2)

    service.complete(
        request_name="test_stage",
        model="anthropic/claude-3.5-haiku",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess1",
        run_id="run1",
    )

    assert len(store.llm_calls) == 1
    call = store.llm_calls[0]
    assert call.session_id == "sess1"
    assert call.run_id == "run1"
    assert call.stage == "test_stage"
    assert call.model == "anthropic/claude-3.5-haiku"
    assert call.output_code == 200
    assert call.schema_version == 2
    assert call.prompt == [{"role": "user", "content": "hi"}]


def test_complete_fallback_path_emits_llm_call_with_none_status():
    store = CaptureMonitoringStore()
    service = LLMService(DummyClientNoRaw(), event_store=store)

    service.complete(
        request_name="fallback_stage",
        model="anthropic/claude-3.5-haiku",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": "hey"}],
        session_id="s",
        run_id="r",
    )

    assert len(store.llm_calls) == 1
    assert store.llm_calls[0].output_code is None
    assert store.llm_calls[0].stage == "fallback_stage"


def test_complete_does_not_emit_llm_call_to_plain_event_store():
    """When the store is a plain NullEventStore (not MonitoringStore), no record_llm_call."""
    store = NullEventStore()
    service = LLMService(DummyClient(), event_store=store)

    # Should not raise even though NullEventStore has record_llm_call as a no-op
    service.complete(
        request_name="test",
        model="anthropic/claude-3.5-haiku",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": "hi"}],
    )
    # No assertion needed — NullEventStore.query_llm_calls returns [] by definition
    assert store.query_llm_calls() == []


def test_complete_with_tools_emits_llm_call_to_monitoring_store():
    store = CaptureMonitoringStore()
    service = LLMService(DummyClient(), event_store=store)

    service.complete_with_tools(
        request_name="agent_stage",
        model="anthropic/claude-3.5-haiku",
        max_tokens=20,
        temperature=0.2,
        messages=[{"role": "user", "content": "build portfolio"}],
        tools=[],
        session_id="sess2",
        run_id="run2",
    )

    assert len(store.llm_calls) == 1
    call = store.llm_calls[0]
    assert call.session_id == "sess2"
    assert call.run_id == "run2"
    assert call.stage == "agent_stage"
    assert call.output_code == 200

