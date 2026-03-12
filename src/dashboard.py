"""Dashboard module for the YFinance Agent application.

Data is fetched from Massive.com (formerly Polygon.io) via the ``massive``
Python SDK.
"""

from __future__ import annotations

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
from src.event_store import create_event_store
from src.llm_validation import (
    extract_valid_tickers,
)
from src.logging_config import set_log_context
from src.plots import (
    plot_history,
    plot_portfolio_allocation,
    plot_portfolio_returns,
)
from src.summaries import (
    build_portfolio_returns_series,
    summarize_portfolio_financials,
    summarize_portfolio_stats,
)
from src.llm_service import LLMService
from src.portfolio_display_summary import PortfolioDisplaySummary
from src.tickr_data_manager import TickrDataManager
from src.tickr_summary_manager import TickrSummaryManager


logger = logging.getLogger(__name__)
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)?$")
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
    logger.info(
        "[session=%s run=%s] [%s] OpenRouter request start model=%s valid_model_format=%s max_tokens=%s temperature=%s messages=%s",
        session_id or "n/a",
        run_id or "n/a",
        request_name,
        model,
        model_valid,
        max_tokens,
        temperature,
        len(messages),
    )
    if not model_valid:
        logger.warning(
            "[session=%s run=%s] [%s] Model name may be malformed: %s",
            session_id or "n/a",
            run_id or "n/a",
            request_name,
            model,
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
        logger.info(
            "[session=%s run=%s] [%s] OpenRouter response received status_code=%s",
            session_id or "n/a",
            run_id or "n/a",
            request_name,
            status_code,
        )
        return raw_response.parse(), status_code
    except AttributeError:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        logger.info(
            "[session=%s run=%s] [%s] OpenRouter response received status_code=unavailable",
            session_id or "n/a",
            run_id or "n/a",
            request_name,
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
    state.setdefault("recommended_tickers", [])
    state.setdefault("excluded_tickers", [])
    state.setdefault("tickr_data_manager", TickrDataManager())
    state.setdefault("tickr_summary_manager", TickrSummaryManager())
    state.setdefault("event_store", None)


def _chat_avatar(role: str) -> Optional[str]:
    if role == "assistant":
        return "img/bot.png"
    return None


def _push_chat_message(role: str, content: str, container) -> None:
    st.session_state["messages"].append({"role": role, "content": content})
    with container:
        with st.chat_message(role, avatar=_chat_avatar(role)):
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
    st.session_state["recommended_tickers"] = list(orchestrator_state.recommended_tickers)
    st.session_state["excluded_tickers"] = list(orchestrator_state.excluded_tickers)

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
        model_labels = openrouter_cfg.get("model_labels", {})
        default_models = openrouter_cfg.get("default_models", {})
        st.sidebar.subheader(ui.get("model_selection_header", "Model Selection"))
        for task_key in ("ticker", "weights", "analysis", "evaluator"):
            label = model_labels.get(task_key, task_key.title())
            default = default_models.get(task_key, model_choices[0])
            idx = model_choices.index(default) if default in model_choices else 0
            st.sidebar.selectbox(
                label,
                options=model_choices,
                index=idx,
                key=f"model_{task_key}",
            )

    api_cfg = openrouter_cfg["api"]
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
        starter_prompt_input = None
        starter_prompts = ui.get("starter_prompts", DEFAULT_STARTER_PROMPTS)
        if starter_prompts:
            st.caption(ui.get("starter_prompts_label", "Try one:"))
            prompt_columns = st.columns(len(starter_prompts))
            for idx, column in enumerate(prompt_columns):
                with column:
                    if st.button(starter_prompts[idx], key=f"starter_prompt_{idx}"):
                        starter_prompt_input = starter_prompts[idx]

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"], avatar=_chat_avatar(message["role"])):
                st.markdown(message["content"])

        typed_prompt_input = st.chat_input(ui["chat_placeholder"])

    prompt_input = typed_prompt_input or starter_prompt_input

    if prompt_input:
        session_id = _get_session_id()
        run_id = _new_correlation_id()
        set_log_context(session_id=session_id, run_id=run_id)
        _push_chat_message("user", prompt_input, chat_tab)

        progress_holder: Dict[str, Any] = {}
        progress_start_text = ui.get("fetch_progress_start", "Fetching ticker data...")
        progress_ticker_template = ui.get(
            "fetch_progress_ticker", "Fetching {ticker} ({current}/{total})"
        )

        def _on_tickers_ready(tickers: List[str]) -> None:
            _push_chat_message(
                "assistant",
                ui["ticker_reply_template"].format(tickers=", ".join(tickers)),
                chat_tab,
            )
            with chat_tab:
                progress_holder["bar"] = st.progress(0.0, text=progress_start_text)

        def _on_fetch_progress(current: int, total: int, ticker: str) -> None:
            bar = progress_holder.get("bar")
            if bar is not None and total > 0:
                bar.progress(
                    current / total,
                    text=progress_ticker_template.format(
                        ticker=ticker, current=current + 1, total=total,
                    ),
                )

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
        if model_choices:
            for task_key in ("ticker", "weights", "analysis", "evaluator"):
                selected = st.session_state.get(f"model_{task_key}")
                if selected:
                    config["openrouter"]["default_models"][task_key] = selected
        display_summary = PortfolioDisplaySummary()
        creator_agent = PortfolioCreatorAgent(
            llm_service=llm_service,
            config=config,
            massive_client_factory=create_massive_client,
            stock_data_fetcher=fetch_stock_data,
            tickr_data_manager=st.session_state["tickr_data_manager"],
            tickr_summary_manager=st.session_state["tickr_summary_manager"],
        )
        evaluator_agent = PortfolioEvaluatorAgent(llm_service=llm_service, config=config)
        orchestrator = AgentOrchestrator(
            creator_agent=creator_agent,
            evaluator_agent=evaluator_agent,
            max_iterations=config.get("agents", {}).get("max_iterations", 3),
            event_store=st.session_state["event_store"],
            schema_version=int(config.get("event_store", {}).get("schema_version", 1)),
        )
        st.session_state["orchestrator"] = orchestrator

        try:
            orchestrator_state = orchestrator.start(
                user_input=prompt_input,
                portfolio_size=st.session_state["portfolio_size"],
                session_id=session_id,
                run_id=run_id,
                ticker_callback=_on_tickers_ready,
                progress_callback=_on_fetch_progress,
            )
            bar = progress_holder.get("bar")
            if bar is not None:
                bar.progress(1.0, text=progress_start_text)
                bar.empty()
        except ValueError as exc:
            bar = progress_holder.get("bar")
            if bar is not None:
                bar.empty()
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

        st.session_state["orchestrator_state"] = orchestrator_state
        _apply_orchestrator_state(orchestrator_state)

        creator_result = orchestrator_state.creator_result
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
                display_summary.format_suggestions(orchestrator_state.pending_suggestions),
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
                if updated_state.pending_suggestions:
                    _push_chat_message(
                        "assistant",
                        PortfolioDisplaySummary().format_suggestions(updated_state.pending_suggestions),
                        chat_tab,
                    )
                if updated_state.status == STATUS_MAX_ITERATIONS_REACHED:
                    _push_chat_message(
                        "assistant",
                        ui.get("evaluator_max_reached", "Reached max refinement iterations. Keeping current portfolio."),
                        chat_tab,
                    )
            elif reject:
                final_state = orchestrator.reject_changes(
                    orchestrator_state,
                    session_id=_get_session_id(),
                    run_id=_new_correlation_id(),
                )
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
            st.write(PortfolioDisplaySummary().format_portfolio_header(tickers))
            st.write(
                f"{ui.get('recommended_label', 'Recommended Tickers:')} "
                f"{st.session_state.get('recommended_tickers', [])}"
            )
            st.write(
                f"{ui.get('excluded_label', 'Tickers to Exclude:')} "
                f"{st.session_state.get('excluded_tickers', [])}"
            )
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

            pending_suggestions = st.session_state.get("pending_suggestions", {})
            if pending_suggestions:
                st.subheader(ui.get("evaluator_changes_label", "Suggested portfolio changes"))
                st.write(PortfolioDisplaySummary().format_suggestions(pending_suggestions))