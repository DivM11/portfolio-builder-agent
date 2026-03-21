"""Unit tests for the Context class that manages agent state."""

from __future__ import annotations

import json

from src.agent_models import AgentContext, AgentResult, Context


def test_context_init_sets_fields() -> None:
    ctx = Context(
        user_input="growth portfolio",
        portfolio_size=1000.0,
        excluded_tickers=["TSLA"],
        session_id="s1",
        run_id="r1",
    )

    assert ctx.user_input == "growth portfolio"
    assert ctx.portfolio_size == 1000.0
    assert ctx.excluded_tickers == ["TSLA"]
    assert ctx.session_id == "s1"
    assert ctx.run_id == "r1"
    assert ctx.messages == []
    assert ctx.work_state == {}
    assert ctx.last_result is None
    assert ctx.tool_invocations == []


def test_context_default_excluded_tickers_empty() -> None:
    ctx = Context(user_input="u", portfolio_size=100.0)

    assert ctx.excluded_tickers == []
    assert ctx.session_id is None
    assert ctx.run_id is None


def test_add_message_appends_to_messages() -> None:
    ctx = Context(user_input="u", portfolio_size=100.0)

    ctx.add_message("system", "You are a portfolio agent.")
    ctx.add_message("user", "Build me a portfolio")

    assert len(ctx.messages) == 2
    assert ctx.messages[0] == {"role": "system", "content": "You are a portfolio agent."}
    assert ctx.messages[1] == {"role": "user", "content": "Build me a portfolio"}


def test_add_assistant_tool_calls_message() -> None:
    ctx = Context(user_input="u", portfolio_size=100.0)
    tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "gen", "arguments": "{}"}}]

    ctx.add_assistant_tool_calls_message("reasoning text", tool_calls)

    assert len(ctx.messages) == 1
    msg = ctx.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "reasoning text"
    assert msg["tool_calls"] == tool_calls


def test_add_tool_result_message() -> None:
    ctx = Context(user_input="u", portfolio_size=100.0)

    ctx.add_tool_result_message("tc1", "generate_tickers", {"tickers": ["AAPL"]})

    assert len(ctx.messages) == 1
    msg = ctx.messages[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "tc1"
    assert msg["name"] == "generate_tickers"
    assert json.loads(msg["content"]) == {"tickers": ["AAPL"]}


def test_update_work_state_merges_keys() -> None:
    ctx = Context(user_input="u", portfolio_size=100.0)

    ctx.update_work_state(tickers=["AAPL"], weights={"AAPL": 1.0})
    ctx.update_work_state(summary="some summary")

    assert ctx.work_state["tickers"] == ["AAPL"]
    assert ctx.work_state["weights"] == {"AAPL": 1.0}
    assert ctx.work_state["summary"] == "some summary"


def test_record_tool_invocation() -> None:
    ctx = Context(user_input="u", portfolio_size=100.0)

    ctx.record_tool_invocation("generate_tickers", {"tickers": ["AAPL"]})

    assert len(ctx.tool_invocations) == 1
    assert ctx.tool_invocations[0] == {"name": "generate_tickers", "arguments": {"tickers": ["AAPL"]}}


def test_to_agent_context_creates_frozen_snapshot() -> None:
    ctx = Context(
        user_input="u",
        portfolio_size=1000.0,
        excluded_tickers=["TSLA"],
        session_id="s1",
        run_id="r1",
    )

    snapshot = ctx.to_agent_context()

    assert isinstance(snapshot, AgentContext)
    assert snapshot.user_input == "u"
    assert snapshot.portfolio_size == 1000.0
    assert snapshot.excluded_tickers == ("TSLA",)
    assert snapshot.session_id == "s1"
    assert snapshot.run_id == "r1"


def test_init_work_state_fresh() -> None:
    ctx = Context(user_input="u", portfolio_size=100.0)

    ctx.init_work_state()

    assert ctx.work_state == {
        "tickers": [],
        "weights": {},
        "allocation": {},
        "summary": "",
        "analysis": {},
        "tool_invocations": [],
        "reasoning_text": "",
    }


def test_init_work_state_from_seed_result() -> None:
    seed = AgentResult(
        tickers=["AAPL", "MSFT"],
        weights={"AAPL": 0.6, "MSFT": 0.4},
        allocation={"AAPL": 600, "MSFT": 400},
        summary_text="summary",
        metadata={"excluded_tickers": ["TSLA"]},
    )
    ctx = Context(user_input="u", portfolio_size=100.0)

    ctx.init_work_state(seed_result=seed)

    assert ctx.work_state["tickers"] == ["AAPL", "MSFT"]
    assert ctx.work_state["weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    assert ctx.work_state["allocation"] == {"AAPL": 600, "MSFT": 400}
    assert ctx.work_state["summary"] == "summary"


def test_prepare_for_run_sets_messages_and_work_state() -> None:
    ctx = Context(user_input="growth", portfolio_size=1000.0)

    ctx.prepare_for_run("You are a portfolio agent.")

    assert len(ctx.messages) == 2
    assert ctx.messages[0]["role"] == "system"
    assert ctx.messages[1] == {"role": "user", "content": "growth"}
    assert ctx.work_state["tickers"] == []


def test_prepare_for_refine_appends_feedback() -> None:
    ctx = Context(user_input="initial", portfolio_size=1000.0)
    ctx.prepare_for_run("system")
    previous = AgentResult(
        tickers=["AAPL"],
        weights={"AAPL": 1.0},
        allocation={"AAPL": 1000},
        metadata={"excluded_tickers": ["TSLA"]},
    )
    ctx.last_result = previous

    ctx.prepare_for_refine("add more tech", previous)

    assert ctx.user_input == "add more tech"
    assert ctx.messages[-1] == {"role": "user", "content": "add more tech"}
    assert ctx.work_state["tickers"] == ["AAPL"]
