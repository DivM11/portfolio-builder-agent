"""Shared LLM service for OpenRouter/OpenAI compatible calls."""

from __future__ import annotations

import logging
import re
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.event_store.base import EventStore, NullEventStore
from src.event_store.models import EventRecord
from openai import OpenAI

logger = logging.getLogger(__name__)
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)?$")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResponse:
    text: str | None
    tool_calls: list[ToolCall]
    has_tool_calls: bool
    raw_response: Any
    usage: dict[str, Any] | None
    status_code: int | None


class LLMService:
    """Lightweight wrapper for OpenRouter chat completions."""

    def __init__(self, client: OpenAI, event_store: EventStore | None = None, schema_version: int = 1) -> None:
        self.client = client
        self.event_store = event_store or NullEventStore()
        self.schema_version = schema_version

    @staticmethod
    def is_model_name_valid(model_name: str) -> bool:
        return bool(MODEL_NAME_PATTERN.match(model_name))

    def complete(
        self,
        *,
        request_name: str,
        model: str,
        max_tokens: int,
        temperature: float,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> tuple[Any, Optional[int]]:
        session = session_id or "n/a"
        run = run_id or "n/a"
        model_valid = self.is_model_name_valid(model)
        start_time = time.perf_counter()
        logger.info(
            "[session=%s run=%s] [%s] OpenRouter request start model=%s valid_model_format=%s max_tokens=%s temperature=%s messages=%s",
            session,
            run,
            request_name,
            model,
            model_valid,
            max_tokens,
            temperature,
            len(messages),
        )
        self.event_store.record(
            EventRecord(
                event_type="llm_request",
                schema_version=self.schema_version,
                session_id=session,
                run_id=run,
                request_name=request_name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
        )
        if not model_valid:
            logger.warning(
                "[session=%s run=%s] [%s] Model name may be malformed: %s",
                session,
                run,
                request_name,
                model,
            )

        try:
            raw_client = self.client.chat.completions.with_raw_response
            raw_response = raw_client.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            status_code = getattr(raw_response, "status_code", None)
            parsed_response = raw_response.parse()
            output = self.extract_message_text(parsed_response)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "[session=%s run=%s] [%s] OpenRouter response received status_code=%s",
                session,
                run,
                request_name,
                status_code,
            )
            self.event_store.record(
                EventRecord(
                    event_type="llm_response",
                    schema_version=self.schema_version,
                    session_id=session,
                    run_id=run,
                    request_name=request_name,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    raw_output=output,
                    token_usage=extract_usage(parsed_response),
                )
            )
            return parsed_response, status_code
        except AttributeError:
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            output = self.extract_message_text(response)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "[session=%s run=%s] [%s] OpenRouter response received status_code=unavailable",
                session,
                run,
                request_name,
            )
            self.event_store.record(
                EventRecord(
                    event_type="llm_response",
                    schema_version=self.schema_version,
                    session_id=session,
                    run_id=run,
                    request_name=request_name,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    status_code=None,
                    latency_ms=latency_ms,
                    raw_output=output,
                    token_usage=extract_usage(response),
                )
            )
            return response, None
        except Exception:
            logger.exception(
                "[session=%s run=%s] [%s] OpenRouter request failed",
                session_id or "n/a",
                run_id or "n/a",
                request_name,
            )
            raise

    def complete_with_tools(
        self,
        *,
        request_name: str,
        model: str,
        max_tokens: int,
        temperature: float,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ToolResponse:
        session = session_id or "n/a"
        run = run_id or "n/a"
        model_valid = self.is_model_name_valid(model)
        start_time = time.perf_counter()
        self.event_store.record(
            EventRecord(
                event_type="llm_request",
                schema_version=self.schema_version,
                session_id=session,
                run_id=run,
                request_name=request_name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
        )
        if not model_valid:
            logger.warning(
                "[session=%s run=%s] [%s] Model name may be malformed: %s",
                session,
                run,
                request_name,
                model,
            )

        try:
            raw_client = self.client.chat.completions.with_raw_response
            raw_response = raw_client.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            status_code = getattr(raw_response, "status_code", None)
            parsed_response = raw_response.parse()
        except AttributeError:
            parsed_response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            status_code = None

        tool_calls = _extract_tool_calls(parsed_response)
        text = self.extract_message_text(parsed_response)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.event_store.record(
            EventRecord(
                event_type="llm_response",
                schema_version=self.schema_version,
                session_id=session,
                run_id=run,
                request_name=request_name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                status_code=status_code,
                latency_ms=latency_ms,
                raw_output=text,
                token_usage=extract_usage(parsed_response),
            )
        )
        return ToolResponse(
            text=text,
            tool_calls=tool_calls,
            has_tool_calls=bool(tool_calls),
            raw_response=parsed_response,
            usage=extract_usage(parsed_response),
            status_code=status_code,
        )

    @staticmethod
    def extract_message_text(response: Any) -> str:
        return extract_message_text(response)


def create_openrouter_client(
    api_key: str,
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    """Create an OpenRouter client using the OpenAI-compatible API."""
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or {})


def build_prompt(template: str, user_input: str, **kwargs: object) -> str:
    """Build the LLM prompt from a template."""
    return template.format(user_input=user_input, **kwargs)


def extract_message_text(response: Any) -> str:
    """Extract text content from an OpenAI-compatible response."""
    try:
        content = response.choices[0].message.content
    except AttributeError:
        content = response["choices"][0]["message"]["content"]
    return content or ""


def extract_usage(response: Any) -> Dict[str, Any] | None:
    """Extract token usage payload if available."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage

    if isinstance(response, dict):
        raw_usage = response.get("usage")
        if isinstance(raw_usage, dict):
            return raw_usage
    return None


def _extract_tool_calls(response: Any) -> list[ToolCall]:
    message = None
    try:
        message = response.choices[0].message
    except AttributeError:
        message = response.get("choices", [{}])[0].get("message", {})

    raw_tool_calls = getattr(message, "tool_calls", None)
    if raw_tool_calls is None and isinstance(message, dict):
        raw_tool_calls = message.get("tool_calls", [])
    if not raw_tool_calls:
        return []

    parsed: list[ToolCall] = []
    for item in raw_tool_calls:
        call_id = getattr(item, "id", None) if not isinstance(item, dict) else item.get("id")
        func = getattr(item, "function", None) if not isinstance(item, dict) else item.get("function", {})
        name = getattr(func, "name", None) if not isinstance(func, dict) else func.get("name")
        arguments_raw = getattr(func, "arguments", "{}") if not isinstance(func, dict) else func.get("arguments", "{}")
        try:
            arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else dict(arguments_raw)
        except Exception:
            arguments = {}
        if name:
            parsed.append(ToolCall(id=call_id or "", name=name, arguments=arguments))
    return parsed
