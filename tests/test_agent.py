"""Unit tests for single portfolio agent behavior."""

from __future__ import annotations

import json
import pandas as pd

from src.agent import PortfolioAgent
from src.agent_models import AgentContext
from src.event_store.base import EventStore
from src.event_store.models import EventRecord


class CaptureEventStore(EventStore):
    def __init__(self) -> None:
        self.events: list[EventRecord] = []

    def record(self, event: EventRecord) -> None:
        self.events.append(event)

    def query(self, **_kwargs):
        return self.events

    def close(self) -> None:
        return None


class DummyLLMService:
    def complete_with_tools(self, **_kwargs):
        raise AssertionError("not needed for these unit tests")


class CaptureLLMService:
    def __init__(self) -> None:
        self.calls = []

    def complete_with_tools(self, **kwargs):
        self.calls.append(kwargs)

        class _Response:
            text = json.dumps(
                {
                    "tickers": ["AAPL", "MSFT"],
                    "weights": {"AAPL": 0.6, "MSFT": 0.4},
                    "allocation": {"AAPL": 600, "MSFT": 400},
                    "analysis_text": "ok",
                    "suggestions": {},
                }
            )
            tool_calls = []
            has_tool_calls = False

        return _Response()


def _base_config() -> dict:
    return {
        "stocks": {
            "max_tickers": 10,
            "history_period": "1y",
            "financials_period": "quarterly",
            "financials_metrics": ["Total Revenue"],
        },
        "event_store": {"schema_version": 2},
        "massive": {"api": {"api_key": "k"}},
    }


def test_parse_final_result_adds_fallback_reweight_suggestion_when_missing() -> None:
    agent = PortfolioAgent(DummyLLMService(), _base_config())
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "weights": {"AAPL": 0.9, "MSFT": 0.1},
        "allocation": {"AAPL": 900, "MSFT": 100},
        "analysis_text": "Concentrated portfolio",
    }

    # preload data cache used by result summary path
    agent.tickr_data_manager.cache = {
        "AAPL": {
            "history": pd.DataFrame({"Close": [1.0, 1.1]}),
            "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
        },
        "MSFT": {
            "history": pd.DataFrame({"Close": [1.0, 1.05]}),
            "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
        },
    }

    result = agent._parse_final_result(json.dumps(payload))

    assert "reweight" in result.suggestions
    assert result.suggestions["reweight"]


def test_execute_tool_logs_tool_call_event_with_explicit_fields() -> None:
    store = CaptureEventStore()
    agent = PortfolioAgent(DummyLLMService(), _base_config(), event_store=store)
    context = AgentContext(user_input="u", portfolio_size=1000.0, session_id="s", run_id="r")

    payload = agent._execute_tool(
        "generate_tickers",
        {"tickers": ["AAPL", "MSFT"]},
        context=context,
        progress_callback=None,
        massive_client=object(),
        work_state={},
    )

    assert payload["valid_tickers"] == ["AAPL", "MSFT"]
    tool_events = [event for event in store.events if event.event_type == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0].tool_name == "generate_tickers"
    assert tool_events[0].tool_arguments == {"tickers": ["AAPL", "MSFT"]}


def test_execute_tool_analyze_builds_summary_if_missing() -> None:
    agent = PortfolioAgent(DummyLLMService(), _base_config())
    agent.tickr_data_manager.cache = {
        "AAPL": {
            "history": pd.DataFrame({"Close": [1.0, 1.1]}),
            "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
        },
        "MSFT": {
            "history": pd.DataFrame({"Close": [1.0, 1.05]}),
            "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
        },
    }
    context = AgentContext(user_input="u", portfolio_size=1000.0, session_id="s", run_id="r")
    work_state = {"summary": "", "tickers": ["AAPL", "MSFT"], "weights": {}, "allocation": {}, "analysis": {}, "tool_invocations": [], "reasoning_text": ""}

    payload = agent._execute_tool(
        "analyze_portfolio",
        {"tickers": ["AAPL", "MSFT"], "weights": {"AAPL": 0.6, "MSFT": 0.4}},
        context=context,
        progress_callback=None,
        massive_client=object(),
        work_state=work_state,
    )

    assert "stats" in payload
    assert work_state["summary"]


def test_agent_run_passes_reasoning_config_to_llm() -> None:
    cfg = _base_config()
    cfg["agent"] = {"model": "anthropic/claude-sonnet-4.5", "max_tokens": 128, "temperature": 0.2, "max_tool_rounds": 1, "reasoning": {"effort": "high", "exclude": False}}
    llm = CaptureLLMService()

    agent = PortfolioAgent(
        llm,
        cfg,
        massive_client_factory=lambda _k: object(),
        stock_data_fetcher=lambda **_kwargs: {},
    )
    agent.tickr_data_manager.cache = {
        "AAPL": {"history": pd.DataFrame({"Close": [1.0]}), "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"])} ,
        "MSFT": {"history": pd.DataFrame({"Close": [1.0]}), "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"])} ,
    }

    agent.run(user_input="u", portfolio_size=1000.0)

    assert llm.calls
    assert llm.calls[0].get("reasoning") == {"effort": "high", "exclude": False}
