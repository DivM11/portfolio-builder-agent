"""Streamlit dashboard for the single-agent portfolio workflow."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.agent import PortfolioAgent
from src.agent_models import AgentResult
from src.data_client import create_massive_client, fetch_stock_data
from src.event_store import create_event_store
from src.llm_service import LLMService, create_openrouter_client
from src.logging_config import set_log_context
from src.plots import plot_history, plot_portfolio_allocation, plot_portfolio_returns
from src.portfolio import allocate_portfolio_by_weights, normalize_weights
from src.portfolio_display_summary import PortfolioDisplaySummary
from src.summaries import (
    build_portfolio_returns_series,
    summarize_portfolio_financials,
    summarize_portfolio_stats,
)
from src.tickr_data_manager import TickrDataManager
from src.tickr_summary_manager import TickrSummaryManager

logger = logging.getLogger(__name__)


DEFAULT_STARTER_PROMPTS = [
    "Build a long-term growth portfolio focused on AI and cloud leaders.",
    "Create a dividend-oriented portfolio for steady income.",
    "Suggest a low-volatility portfolio across defensive sectors.",
    "Make a balanced portfolio with large-cap US equities.",
]


def _new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def _get_session_id() -> str:
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = _new_correlation_id()
        st.session_state["session_id"] = session_id
    return session_id


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def _chat_avatar(role: str) -> Optional[str]:
    if role == "assistant":
        return "img/bot.png"
    return None


def _stream_once(content: str):
    yield content


def _push_chat_message(role: str, content: str, container) -> None:
    st.session_state["messages"].append({"role": role, "content": content})
    with container:
        with st.chat_message(role, avatar=_chat_avatar(role)):
            if role == "assistant" and hasattr(st, "write_stream"):
                st.write_stream(_stream_once(content))
            else:
                st.markdown(content)


def _strip_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text.strip()
    return (text[:start] + text[end + 1 :]).strip()


def _clean_reasoning_text(reasoning_text: str) -> str:
    cleaned = _strip_json_block(reasoning_text)
    # Remove any leftover structured-key mentions from text output.
    cleaned = re.sub(r'"?(tickers|weights|allocation)"?\s*:\s*[^\n]+', "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _build_allocation_table(tickers: list[str], weights: dict[str, float], allocation: dict[str, float]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        rows.append(
            {
                "Ticker": ticker,
                "Weight": float(weights.get(ticker, 0.0)),
                "Allocation": float(allocation.get(ticker, 0.0)),
            }
        )
    
    return pd.DataFrame(rows)


def _format_suggestions_text(suggestions: dict[str, Any], weights: dict[str, float]) -> str:
    if suggestions:
        return PortfolioDisplaySummary().format_suggestions(suggestions)
    if weights:
        return PortfolioDisplaySummary().format_suggestions({"add": [], "remove": [], "reweight": weights})
    return "No suggested changes."


def _render_current_portfolio_sections(container, ui: Dict[str, Any]) -> None:
    allocation_df = st.session_state.get("allocation_table_df")
    if isinstance(allocation_df, pd.DataFrame) and not allocation_df.empty:
        with container:
            st.subheader(ui.get("current_portfolio_table_label", "Current Portfolio Allocation"))
            st.dataframe(
                allocation_df,
                column_config={
                    "Weight": st.column_config.NumberColumn(
                        "Weight (%)",
                        help="Weight of the ticker in the portfolio",
                        format="%.2f %%" # Use '%%' to display a literal '%' sign
                    )},
                 width="stretch", hide_index=True)

            st.subheader(ui.get("analysis_heading", "Portfolio Analysis"))
            st.write(st.session_state.get("analysis_text", ""))

            st.subheader(ui.get("suggestions_heading", "Suggested portfolio changes"))
            st.write(st.session_state.get("suggestions_text", "No suggested changes."))

            reasoning_display = st.session_state.get("reasoning_display_text", "")
            if reasoning_display:
                with st.expander(ui.get("reasoning_expander_label", "Reasoning (hidden by default)"), expanded=False):
                    st.markdown(reasoning_display)


def _init_state(default_user_input: str, default_portfolio_size: float, chat_intro: str) -> None:
    state = st.session_state
    state.setdefault("messages", [{"role": "assistant", "content": chat_intro}])
    state.setdefault("user_input", default_user_input)
    state.setdefault("tickers", [])
    state.setdefault("data_by_ticker", {})
    state.setdefault("portfolio_size", default_portfolio_size)
    state.setdefault("weights", {})
    state.setdefault("portfolio_allocation", {})
    state.setdefault("portfolio_stats", {})
    state.setdefault("portfolio_financials", {})
    state.setdefault("portfolio_series", pd.Series(dtype=float))
    state.setdefault("analysis_text", "")
    state.setdefault("reasoning_display_text", "")
    state.setdefault("suggestions_text", "")
    state.setdefault("allocation_table_df", pd.DataFrame())
    state.setdefault("pending_suggestions", {})
    state.setdefault("recommended_tickers", [])
    state.setdefault("excluded_tickers", [])
    state.setdefault("tickr_data_manager", TickrDataManager())
    state.setdefault("tickr_summary_manager", TickrSummaryManager())
    state.setdefault("event_store", None)
    state.setdefault("portfolio_agent", None)
    state.setdefault("latest_result", AgentResult())
    state.setdefault("is_processing", False)
    state.setdefault("pending_prompt", None)
    state.setdefault("awaiting_user_decision", False)


def _apply_agent_result(result: AgentResult, financial_metrics: list[str]) -> None:
    st.session_state["tickers"] = result.tickers
    st.session_state["data_by_ticker"] = result.data_by_ticker
    weights = dict(result.weights)
    if not weights and result.allocation:
        total = float(sum(result.allocation.values()))
        if total > 0:
            weights = {ticker: float(amount) / total for ticker, amount in result.allocation.items()}

    tickers = list(result.tickers)
    normalized_weights = normalize_weights(weights, tickers)

    portfolio_size = float(st.session_state.get("portfolio_size", 0.0))
    allocation = dict(result.allocation)
    total_allocation = float(sum(allocation.values())) if allocation else 0.0
    has_valid_allocation = allocation and portfolio_size > 0 and abs(total_allocation - portfolio_size) <= max(1.0, portfolio_size * 0.01)
    if not has_valid_allocation:
        allocation = allocate_portfolio_by_weights(
            tickers,
            portfolio_size,
            normalized_weights,
        )

    st.session_state["weights"] = normalized_weights
    st.session_state["portfolio_allocation"] = allocation
    st.session_state["analysis_text"] = result.analysis_text
    st.session_state["pending_suggestions"] = result.suggestions
    st.session_state["latest_result"] = result
    st.session_state["suggestions_text"] = _format_suggestions_text(result.suggestions, normalized_weights)
    st.session_state["allocation_table_df"] = _build_allocation_table(tickers, normalized_weights, allocation)

    reasoning_text = str(result.metadata.get("reasoning_text", "")).strip()
    st.session_state["reasoning_display_text"] = _clean_reasoning_text(reasoning_text) if reasoning_text else ""

    portfolio_series = build_portfolio_returns_series(
        {ticker: data["history"] for ticker, data in result.data_by_ticker.items()},
        normalized_weights,
    )
    st.session_state["portfolio_series"] = portfolio_series
    st.session_state["portfolio_stats"] = summarize_portfolio_stats(portfolio_series)
    st.session_state["portfolio_financials"] = summarize_portfolio_financials(
        {ticker: data["financials"] for ticker, data in result.data_by_ticker.items()},
        normalized_weights,
        financial_metrics,
    )


def run_dashboard(config: Dict[str, Any]) -> None:
    st.title(config["app"]["title"])

    ui = config["ui"]
    dashboard = config["dashboard"]
    openrouter_cfg = config["openrouter"]
    stocks_cfg = config["stocks"]
    financial_metrics = stocks_cfg.get("financials_metrics", [])

    _init_state(dashboard["default_user_input"], dashboard["default_portfolio_size"], ui["chat_intro"])
    session_id = _get_session_id()
    set_log_context(session_id=session_id)
    if st.session_state.get("event_store") is None:
        st.session_state["event_store"] = create_event_store(config.get("event_store", {}))

    st.sidebar.header(ui["sidebar_header"])
    st.sidebar.number_input(
        ui["portfolio_size_label"],
        min_value=0.0,
        step=100.0,
        key="portfolio_size",
    )

    model_choices = openrouter_cfg.get("model_choices", [])
    if model_choices:
        agent_cfg = config.setdefault("agent", {})
        current_model = agent_cfg.get("model", model_choices[0])
        idx = model_choices.index(current_model) if current_model in model_choices else 0
        selected_model = st.sidebar.selectbox(
            "Agent Model",
            options=model_choices,
            index=idx,
            key="model_agent",
        )
        config.setdefault("agent", {})["model"] = selected_model

    api_cfg = openrouter_cfg["api"]
    api_key = api_cfg.get("api_key")
    if not api_key:
        st.sidebar.error(ui["missing_api_key"])
        return

    massive_cfg = config.get("massive", {}).get("api", {})
    if not massive_cfg.get("api_key"):
        st.sidebar.error(ui.get("missing_massive_key", "Missing Massive.com API key."))
        return

    tabs = st.tabs([ui["chat_tab_label"], ui["history_tab_label"], ui["portfolio_tab_label"]])
    chat_tab, history_tab, portfolio_tab = tabs

    with chat_tab:
        starter_prompt_input = None
        starter_prompts = ui.get("starter_prompts", DEFAULT_STARTER_PROMPTS)
        if starter_prompts and not st.session_state.get("is_processing", False):
            st.caption(ui.get("starter_prompts_label", "Try one:"))
            prompt_columns = st.columns(len(starter_prompts))
            for idx, column in enumerate(prompt_columns):
                with column:
                    if st.button(starter_prompts[idx], key=f"starter_prompt_{idx}"):
                        starter_prompt_input = starter_prompts[idx]

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"], avatar=_chat_avatar(message["role"])):
                st.markdown(message["content"])

        _render_current_portfolio_sections(chat_tab, ui)

        if st.session_state.get("is_processing", False):
            typed_prompt_input = None
        else:
            typed_prompt_input = st.chat_input(ui["chat_placeholder"])

    if typed_prompt_input or starter_prompt_input:
        st.session_state["pending_prompt"] = typed_prompt_input or starter_prompt_input
        st.session_state["is_processing"] = True

    prompt_input = st.session_state.get("pending_prompt") if st.session_state.get("is_processing", False) else None

    if prompt_input:
        run_id = _new_correlation_id()
        set_log_context(session_id=session_id, run_id=run_id)
        _push_chat_message("user", prompt_input, chat_tab)

        client = create_openrouter_client(
            api_key=api_key,
            base_url=api_cfg["base_url"],
            headers={
                "HTTP-Referer": api_cfg["http_referer"],
                "X-Title": api_cfg["app_title"],
            },
        )
        llm_service = LLMService(
            client,
            event_store=st.session_state["event_store"],
            schema_version=int(config.get("event_store", {}).get("schema_version", 1)),
        )
        agent = PortfolioAgent(
            llm_service=llm_service,
            config=config,
            event_store=st.session_state["event_store"],
            tickr_data_manager=st.session_state["tickr_data_manager"],
            tickr_summary_manager=st.session_state["tickr_summary_manager"],
            massive_client_factory=create_massive_client,
            stock_data_fetcher=fetch_stock_data,
        )
        st.session_state["portfolio_agent"] = agent

        progress_holder: Dict[str, Any] = {}
        progress_start_text = ui.get("fetch_progress_start", "Fetching ticker data...")
        progress_ticker_template = ui.get("fetch_progress_ticker", "Fetching {ticker} ({current}/{total})")
        weights_progress_text = ui.get("weights_progress_text", "Computing portfolio weights...")
        analysis_progress_text = ui.get("analysis_progress_text", "Analyzing portfolio...")

        def _on_fetch_progress(current: int, total: int, ticker: str) -> None:
            bar = progress_holder.get("bar")
            if bar is None:
                return
            if total > 0:
                bar.progress(
                    min(((current + 1) / total) * 0.7, 0.7),
                    text=progress_ticker_template.format(ticker=ticker, current=current + 1, total=total),
                )

        def _on_agent_step(step: str) -> None:
            bar = progress_holder.get("bar")
            if bar is None:
                return
            if step == "allocate_weights":
                bar.progress(0.85, text=weights_progress_text)
            elif step == "analyze_portfolio":
                bar.progress(0.95, text=analysis_progress_text)

        try:
            with chat_tab:
                progress_holder["bar"] = st.progress(0.0, text=progress_start_text)
            result = agent.run(
                user_input=prompt_input,
                portfolio_size=st.session_state["portfolio_size"],
                excluded_tickers=list(st.session_state.get("excluded_tickers", [])),
                session_id=session_id,
                run_id=run_id,
                progress_callback=_on_fetch_progress,
                status_callback=_on_agent_step,
            )
            bar = progress_holder.get("bar")
            if bar is not None:
                bar.progress(1.0, text=progress_start_text)
                bar.empty()
        except ValueError as exc:
            bar = progress_holder.get("bar")
            if bar is not None:
                bar.empty()
            st.session_state["pending_prompt"] = None
            st.session_state["is_processing"] = False
            _push_chat_message("assistant", str(exc), chat_tab)
            return

        st.session_state["pending_prompt"] = None
        st.session_state["is_processing"] = False
        _apply_agent_result(result, financial_metrics)
        st.session_state["awaiting_user_decision"] = bool(result.weights or result.suggestions)
        if result.tickers:
            _push_chat_message("assistant", ui["ticker_reply_template"].format(tickers=", ".join(result.tickers)), chat_tab)
        _render_current_portfolio_sections(chat_tab, ui)
        _push_chat_message("assistant", ui["post_analysis_nudge"], chat_tab)
        st.rerun()

    agent = st.session_state.get("portfolio_agent")
    pending_suggestions = st.session_state.get("pending_suggestions", {})
    awaiting_user_decision = bool(st.session_state.get("awaiting_user_decision", False))
    if agent and awaiting_user_decision:
        with chat_tab:
            if not pending_suggestions:
                st.caption("Review the proposed portfolio and choose whether to apply further changes.")
            accept = st.button(ui.get("evaluator_accept_button", "Accept Changes"), key="accept_changes")
            reject = st.button(ui.get("evaluator_reject_button", "Keep Current Portfolio"), key="reject_changes")

        if accept:
            updated = agent.refine(
                feedback="Apply the suggested changes and return the updated portfolio.",
                session_id=session_id,
                run_id=_new_correlation_id(),
            )
            _apply_agent_result(updated, financial_metrics)
            st.session_state["awaiting_user_decision"] = bool(updated.suggestions)
            if updated.analysis_text:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": "Portfolio updated. Review the latest analysis and suggestions below."}
                )
            st.rerun()

        if reject:
            st.session_state["pending_suggestions"] = {}
            st.session_state["awaiting_user_decision"] = False
            st.session_state["messages"].append(
                {"role": "assistant", "content": ui.get("evaluator_rejected", "Keeping current portfolio without changes.")}
            )
            st.rerun()

    tickers = st.session_state.get("tickers", [])
    st.sidebar.write(ui["suggested_label"], tickers)
    data_by_ticker = st.session_state.get("data_by_ticker", {})

    with history_tab:
        if not tickers:
            st.info(ui["history_empty_message"])
        else:
            history_fig = plot_history(
                {ticker: data["history"] for ticker, data in data_by_ticker.items()},
                selected_tickers=tickers,
            )
            if history_fig is not None:
                st.plotly_chart(history_fig, width="stretch")

            st.caption(ui["download_prompt"])
            for ticker in tickers:
                history = data_by_ticker.get(ticker, {}).get("history", pd.DataFrame())
                st.download_button(
                    label=f"{ui['download_history_label']} ({ticker})",
                    data=_df_to_csv_bytes(history),
                    file_name=f"{ticker}_history.csv",
                    mime="text/csv",
                )

    with portfolio_tab:
        if not tickers:
            st.info(ui["portfolio_empty_message"])
        else:
            st.write(PortfolioDisplaySummary().format_portfolio_header(tickers))
            _render_current_portfolio_sections(portfolio_tab, ui)
            allocation = st.session_state.get("portfolio_allocation", {})
            if allocation:
                st.subheader(ui["portfolio_output_label"])
                alloc_fig = plot_portfolio_allocation(allocation, title=ui["portfolio_output_label"])
                if alloc_fig is not None:
                    st.plotly_chart(alloc_fig, width="stretch")

            stats = st.session_state.get("portfolio_stats", {})
            if stats:
                st.subheader(ui["portfolio_stats_label"])
                stats_df = pd.DataFrame(
                    [
                        {
                            "Min": f"{stats.get('min', 0):.2f}",
                            "Max": f"{stats.get('max', 0):.2f}",
                            "Median": f"{stats.get('median', 0):.2f}",
                            "Current": f"{stats.get('current', 0):.2f}",
                            "1Y Return": f"{stats.get('return_1y', 0):.2%}",
                        }
                    ]
                )
                st.dataframe(stats_df, width="stretch", hide_index=True)

            portfolio_series = st.session_state.get("portfolio_series", pd.Series(dtype=float))
            returns_fig = plot_portfolio_returns(portfolio_series, ui["portfolio_returns_label"])
            if returns_fig is not None:
                st.plotly_chart(returns_fig, width="stretch")

            portfolio_financials = st.session_state.get("portfolio_financials", {})
            if portfolio_financials:
                st.subheader(ui["portfolio_financials_label"])
                st.dataframe(pd.DataFrame([portfolio_financials]), width="stretch", hide_index=True)

            pending_suggestions = st.session_state.get("pending_suggestions", {})
            if pending_suggestions:
                st.subheader(ui.get("evaluator_changes_label", "Suggested portfolio changes"))
                st.write(PortfolioDisplaySummary().format_suggestions(pending_suggestions))
