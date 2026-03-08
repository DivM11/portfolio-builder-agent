"""Dashboard module for the YFinance Agent application.

Data is fetched from Massive.com (formerly Polygon.io) via the ``massive``
Python SDK.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
from openai import OpenAI

from src.agents.creator import PortfolioCreatorAgent
from src.agents.evaluator import PortfolioEvaluatorAgent
from src.agents.orchestrator import (
    AgentOrchestrator,
    STATUS_AWAITING_APPROVAL,
    STATUS_MAX_ITERATIONS_REACHED,
)

from src.data_client import (
    create_massive_client,
    fetch_stock_data as _massive_fetch_stock_data,
)
from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_weights_payload,
)
from src.plots import (
    plot_history,
    plot_portfolio_allocation,
    plot_portfolio_returns,
)
from src.portfolio import allocate_portfolio_by_weights, format_portfolio_allocation, normalize_weights
from src.summaries import (
    build_portfolio_returns_series,
    build_portfolio_summary,
    summarize_portfolio_financials,
    summarize_portfolio_stats,
)
from src.llm_service import LLMService


logger = logging.getLogger(__name__)
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)?$")


def _new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def _get_session_id() -> str:
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = _new_correlation_id()
        st.session_state["session_id"] = session_id
    return session_id


def _log_backend(message: str, *args: object, session_id: Optional[str] = None, run_id: Optional[str] = None) -> None:
    prefix = f"[session={session_id or 'n/a'} run={run_id or 'n/a'}] "
    logger.warning(prefix + message, *args)


def _is_model_name_valid(model_name: str) -> bool:
    return bool(MODEL_NAME_PATTERN.match(model_name))


def _create_openrouter_completion(
    *,
    client: OpenAI,
    request_name: str,
    model: str,
    max_tokens: int,
    temperature: float,
    messages: List[Dict[str, str]],
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple[Any, Optional[int]]:
    model_valid = _is_model_name_valid(model)
    _log_backend(
        "[%s] OpenRouter request start model=%s valid_model_format=%s max_tokens=%s temperature=%s messages=%s",
        request_name,
        model,
        model_valid,
        max_tokens,
        temperature,
        len(messages),
        session_id=session_id,
        run_id=run_id,
    )
    if not model_valid:
        _log_backend(
            "[%s] Model name may be malformed: %s",
            request_name,
            model,
            session_id=session_id,
            run_id=run_id,
        )

    try:
        raw_client = client.chat.completions.with_raw_response
        raw_response = raw_client.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        status_code = getattr(raw_response, "status_code", None)
        _log_backend(
            "[%s] OpenRouter response received status_code=%s",
            request_name,
            status_code,
            session_id=session_id,
            run_id=run_id,
        )
        return raw_response.parse(), status_code
    except AttributeError:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        _log_backend(
            "[%s] OpenRouter response received status_code=unavailable",
            request_name,
            session_id=session_id,
            run_id=run_id,
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


def parse_tickers(text: str, delimiter: str) -> List[str]:
    """Parse a delimited ticker list into clean symbols."""
    return extract_valid_tickers(text, delimiter=delimiter)


def fetch_stock_data(
    ticker: str,
    history_period: str,
    financials_period: str,
    massive_client: Any = None,
) -> Dict[str, Any]:
    """Fetch stock data from Massive.com (formerly Polygon.io).

    Parameters
    ----------
    ticker:
        US equity symbol.
    history_period:
        Lookback period string (e.g. ``"1y"``).
    financials_period:
        ``"annual"`` or ``"quarterly"``.
    massive_client:
        An authenticated ``massive.RESTClient``.  If *None*, the caller must
        have already ensured a client is available (used only in tests).
    """
    if massive_client is None:
        raise ValueError("A Massive.com RESTClient must be provided.")
    return _massive_fetch_stock_data(
        client=massive_client,
        ticker=ticker,
        history_period=history_period,
        financials_period=financials_period,
    )


def _extract_message_text(response: Any) -> str:
    """Extract text content from an OpenAI-compatible response."""
    try:
        return response.choices[0].message.content
    except AttributeError:
        return response["choices"][0]["message"]["content"]


def generate_tickers(
    client: OpenAI,
    prompt: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    delimiter: str,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple[List[str], str]:
    """Generate ticker suggestions from the LLM and return parsed + raw output."""
    response, _status_code = _create_openrouter_completion(
        client=client,
        request_name="ticker_generation",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        session_id=session_id,
        run_id=run_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    raw_output = _extract_message_text(response)
    return parse_tickers(raw_output, delimiter=delimiter), raw_output


def limit_tickers(tickers: List[str], max_tickers: int) -> List[str]:
    """Limit the number of tickers returned."""
    if max_tickers <= 0:
        return []
    return tickers[:max_tickers]


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


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
    state.setdefault("orchestrator_state", None)
    state.setdefault("agent_iteration", 0)
    state.setdefault("pending_suggestions", {})
    state.setdefault("orchestrator", None)


def _push_chat_message(role: str, content: str, container) -> None:
    st.session_state["messages"].append({"role": role, "content": content})
    with container:
        with st.chat_message(role):
            st.markdown(content)


def _apply_orchestrator_state(orchestrator_state: Any) -> None:
    creator_result = orchestrator_state.creator_result
    evaluator_result = orchestrator_state.evaluator_result

    st.session_state["tickers"] = creator_result.tickers
    st.session_state["data_by_ticker"] = creator_result.data_by_ticker
    st.session_state["weights"] = creator_result.weights
    st.session_state["portfolio_allocation"] = creator_result.allocation
    st.session_state["analysis_text"] = evaluator_result.analysis_text
    st.session_state["pending_suggestions"] = orchestrator_state.pending_suggestions
    st.session_state["agent_iteration"] = orchestrator_state.iteration

    portfolio_series = build_portfolio_returns_series(
        {ticker: data["history"] for ticker, data in creator_result.data_by_ticker.items()},
        creator_result.weights,
    )
    st.session_state["portfolio_series"] = portfolio_series
    st.session_state["portfolio_stats"] = summarize_portfolio_stats(portfolio_series)
    st.session_state["portfolio_financials"] = summarize_portfolio_financials(
        {ticker: data["financials"] for ticker, data in creator_result.data_by_ticker.items()},
        creator_result.weights,
        st.session_state.get("_financial_metrics", []),
    )


def run_dashboard(config: Dict[str, Any]) -> None:
    """Run the Streamlit dashboard."""
    st.title(config["app"]["title"])

    ui = config["ui"]
    dashboard = config["dashboard"]
    openrouter_cfg = config["openrouter"]
    stocks_cfg = config["stocks"]
    st.session_state["_financial_metrics"] = stocks_cfg.get("financials_metrics", [])

    _init_state(
        dashboard["default_user_input"],
        dashboard["default_portfolio_size"],
        ui["chat_intro"],
    )
    _get_session_id()

    st.sidebar.header(ui["sidebar_header"])
    st.sidebar.number_input(
        ui["portfolio_size_label"],
        min_value=0.0,
        step=100.0,
        key="portfolio_size",
    )

    api_cfg = openrouter_cfg["api"]
    models_cfg = openrouter_cfg["models"]
    outputs_cfg = openrouter_cfg["outputs"]
    temperatures_cfg = openrouter_cfg["temperatures"]
    prompts_cfg = openrouter_cfg["prompts"]

    api_key = api_cfg.get("api_key")
    if not api_key:
        st.sidebar.error(ui["missing_api_key"])
        return

    tabs = st.tabs(
        [
            ui["chat_tab_label"],
            ui["history_tab_label"],
            ui["portfolio_tab_label"],
        ]
    )
    chat_tab, history_tab, portfolio_tab = tabs

    with chat_tab:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt_input = st.chat_input(ui["chat_placeholder"])

    if prompt_input:
        session_id = _get_session_id()
        run_id = _new_correlation_id()
        _push_chat_message("user", prompt_input, chat_tab)
        progress_text = ui.get("fetch_progress_start", "Fetching ticker data...")
        with chat_tab:
            progress = st.progress(0.0, text=progress_text)

        client = create_openrouter_client(
            api_key=api_key,
            base_url=api_cfg["base_url"],
            headers={
                "HTTP-Referer": api_cfg["http_referer"],
                "X-Title": api_cfg["app_title"],
            },
        )
        llm_service = LLMService(client)
        creator_agent = PortfolioCreatorAgent(
            llm_service=llm_service,
            config=config,
            massive_client_factory=create_massive_client,
            stock_data_fetcher=fetch_stock_data,
        )
        evaluator_agent = PortfolioEvaluatorAgent(llm_service=llm_service, config=config)
        orchestrator = AgentOrchestrator(
            creator_agent=creator_agent,
            evaluator_agent=evaluator_agent,
            max_iterations=config.get("agents", {}).get("max_iterations", 3),
        )
        st.session_state["orchestrator"] = orchestrator

        try:
            orchestrator_state = orchestrator.start(
                user_input=prompt_input,
                portfolio_size=st.session_state["portfolio_size"],
                session_id=session_id,
                run_id=run_id,
            )
            progress.progress(1.0, text=progress_text)
        except ValueError as exc:
            progress.empty()
            error_text = str(exc)
            if "No valid ticker symbols" in error_text:
                _push_chat_message("assistant", ui["ticker_validation_error"], chat_tab)
            elif "Rate-limited while fetching historical price data for:" in error_text:
                with chat_tab:
                    st.warning(error_text)
                _push_chat_message("assistant", error_text, chat_tab)
            elif "Could not fetch historical price data for any suggested ticker" in error_text:
                _push_chat_message(
                    "assistant",
                    ui.get(
                        "history_fetch_all_failed",
                        "Could not fetch historical price data for any suggested ticker.",
                    ),
                    chat_tab,
                )
            else:
                _push_chat_message("assistant", error_text, chat_tab)
            return

            progress.empty()

        st.session_state["orchestrator_state"] = orchestrator_state
        _apply_orchestrator_state(orchestrator_state)

        creator_result = orchestrator_state.creator_result
        _push_chat_message(
            "assistant",
            ui["ticker_reply_template"].format(tickers=", ".join(creator_result.tickers)),
            chat_tab,
        )

        warning_templates = {
            "rate_limited": ui.get(
                "history_fetch_warning_rate_limited",
                "Rate-limited while fetching historical price data for: {tickers}. Please retry shortly.",
            ),
            "not_found": ui.get(
                "history_fetch_warning_not_found",
                "Ticker not found for historical price data: {tickers}. These were skipped.",
            ),
            "empty_data": ui.get(
                "history_fetch_warning_empty_data",
                ui.get("history_fetch_warning", "No historical data was found for: {tickers}"),
            ),
            "unexpected_error": ui.get(
                "history_fetch_warning_unexpected_error",
                "Unexpected error while fetching historical price data for: {tickers}. These were skipped.",
            ),
        }
        failed_history_by_status = creator_result.metadata.get("failed_history_by_status", {})
        for status, failed_tickers in failed_history_by_status.items():
            if not failed_tickers:
                continue
            warning_message = warning_templates.get(status, warning_templates["unexpected_error"]).format(
                tickers=", ".join(failed_tickers)
            )
            with chat_tab:
                st.warning(warning_message)
            _push_chat_message("assistant", warning_message, chat_tab)

        if creator_result.metadata.get("weights_parse_failed"):
            _push_chat_message("assistant", ui["weights_fallback_message"], chat_tab)

        dropped = creator_result.metadata.get("weights_dropped", [])
        if dropped:
            _push_chat_message(
                "assistant",
                ui["weights_tickers_dropped"].format(dropped=", ".join(dropped)),
                chat_tab,
            )

        _push_chat_message("assistant", orchestrator_state.evaluator_result.analysis_text, chat_tab)

        if orchestrator_state.pending_suggestions:
            _push_chat_message(
                "assistant",
                f"{ui.get('evaluator_changes_label', 'Suggested portfolio changes')}: {json.dumps(orchestrator_state.pending_suggestions)}",
                chat_tab,
            )
        else:
            _push_chat_message("assistant", ui.get("evaluator_no_changes", "No further portfolio changes suggested."), chat_tab)

        _push_chat_message("assistant", ui["post_analysis_nudge"], chat_tab)
        portfolio_link_message = ui.get("portfolio_tab_link")
        if portfolio_link_message:
            _push_chat_message("assistant", portfolio_link_message, chat_tab)

    orchestrator_state = st.session_state.get("orchestrator_state")
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator_state and orchestrator and orchestrator_state.status == STATUS_AWAITING_APPROVAL:
        if hasattr(st, "button"):
            with chat_tab:
                st.markdown(
                    ui.get("evaluator_iter_label", "Iteration {current} of {max_iterations}").format(
                        current=orchestrator_state.iteration,
                        max_iterations=config.get("agents", {}).get("max_iterations", 3),
                    )
                )
                accept = st.button(ui.get("evaluator_accept_button", "Accept Changes"), key="accept_changes")
                reject = st.button(ui.get("evaluator_reject_button", "Keep Current Portfolio"), key="reject_changes")
            if accept:
                updated_state = orchestrator.apply_changes(
                    orchestrator_state,
                    session_id=_get_session_id(),
                    run_id=_new_correlation_id(),
                )
                st.session_state["orchestrator_state"] = updated_state
                _apply_orchestrator_state(updated_state)
                _push_chat_message("assistant", updated_state.evaluator_result.analysis_text, chat_tab)
                if updated_state.status == STATUS_MAX_ITERATIONS_REACHED:
                    _push_chat_message(
                        "assistant",
                        ui.get("evaluator_max_reached", "Reached max refinement iterations. Keeping current portfolio."),
                        chat_tab,
                    )
            elif reject:
                final_state = orchestrator.reject_changes(orchestrator_state)
                st.session_state["orchestrator_state"] = final_state
                _push_chat_message("assistant", ui.get("evaluator_rejected", "Keeping current portfolio without changes."), chat_tab)

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
            allocation = st.session_state.get("portfolio_allocation", {})
            if allocation:
                st.subheader(ui["portfolio_output_label"])
                alloc_fig = plot_portfolio_allocation(
                    allocation, title=ui["portfolio_output_label"],
                )
                if alloc_fig is not None:
                    st.plotly_chart(alloc_fig, width="stretch")

            stats = st.session_state.get("portfolio_stats", {})
            if stats:
                st.subheader(ui["portfolio_stats_label"])
                stats_df = pd.DataFrame(
                    [{
                        "Min": f"{stats.get('min', 0):.2f}",
                        "Max": f"{stats.get('max', 0):.2f}",
                        "Median": f"{stats.get('median', 0):.2f}",
                        "Current": f"{stats.get('current', 0):.2f}",
                        "1Y Return": f"{stats.get('return_1y', 0):.2%}",
                    }]
                )
                st.dataframe(stats_df, width="stretch", hide_index=True)

            portfolio_series = st.session_state.get("portfolio_series", pd.Series(dtype=float))
            returns_fig = plot_portfolio_returns(portfolio_series, ui["portfolio_returns_label"])
            if returns_fig is not None:
                st.plotly_chart(returns_fig, width="stretch")

            portfolio_financials = st.session_state.get("portfolio_financials", {})
            if portfolio_financials:
                st.subheader(ui["portfolio_financials_label"])
                st.dataframe(
                    pd.DataFrame([portfolio_financials]),
                    width="stretch",
                    hide_index=True,
                )