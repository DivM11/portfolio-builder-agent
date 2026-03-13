"""Unit tests for shared LLM service."""

from src.llm_service import LLMService, build_prompt, extract_message_text


class DummyRawResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def parse(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class DummyWithRaw:
    def create(self, **_kwargs):
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
