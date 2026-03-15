"""Unit tests for the single-agent dashboard module."""

from __future__ import annotations

import pandas as pd

from src.agent_models import AgentResult
from src.dashboard import run_dashboard


class DummyContainer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyPlaceholder:
    def __init__(self, progress_updates):
        self._progress_updates = progress_updates

    def progress(self, value: float, text: str | None = None):
        self._progress_updates.append((value, text))
        return self

    def empty(self):
        return None


class DummySidebar:
    def __init__(self) -> None:
        self.session_state = None
        self.error_msg = None
        self.selectbox_calls = []
        self.write_args = None

    def header(self, _text: str) -> None:
        return None

    def number_input(self, _label: str, min_value: float, step: float, key: str | None = None):
        if key and self.session_state is not None:
            self.session_state.setdefault(key, min_value)
            return self.session_state[key]
        return min_value

    def selectbox(self, label: str, options: list[str], index: int = 0, key: str | None = None):
        self.selectbox_calls.append((label, options, index, key))
        if key and self.session_state is not None:
            self.session_state.setdefault(key, options[index])
            return self.session_state[key]
        return options[index]

    def error(self, msg: str) -> None:
        self.error_msg = msg

    def write(self, label: str, value) -> None:
        self.write_args = (label, value)


class DummyStreamlit:
    def __init__(self, sidebar: DummySidebar, prompt: str | None):
        self.sidebar = sidebar
        self.session_state = {}
        self.sidebar.session_state = self.session_state
        self._prompt = prompt
        self.markdowns: list[str] = []
        self.infos: list[str] = []
        self.progress_updates: list[tuple[float, str | None]] = []
        self.tabs_created = []
        self.plot_kwargs = []
        self.dataframe_kwargs = []
        self.button_calls = []
        self.streamed_messages: list[str] = []

    def title(self, _text: str) -> None:
        return None

    def tabs(self, labels):
        self.tabs_created = labels
        return [DummyContainer() for _ in labels]

    def caption(self, _text: str) -> None:
        return None

    def columns(self, count):
        return [DummyContainer() for _ in range(count)]

    def button(self, _label: str, key: str | None = None):
        self.button_calls.append((key or _label, _label))
        return False

    def chat_input(self, _placeholder: str):
        return self._prompt

    def chat_message(self, _role: str, avatar: str | None = None):
        return DummyContainer()

    def markdown(self, text: str) -> None:
        self.markdowns.append(text)

    def write_stream(self, stream):
        chunks = list(stream)
        text = "".join(str(chunk) for chunk in chunks)
        self.streamed_messages.append(text)
        self.markdowns.append(text)
        return text

    def progress(self, value: float, text: str | None = None):
        self.progress_updates.append((value, text))
        return DummyPlaceholder(self.progress_updates)

    def info(self, text: str) -> None:
        self.infos.append(text)

    def plotly_chart(self, _fig, **kwargs):
        self.plot_kwargs.append(kwargs)

    def dataframe(self, _df, **kwargs):
        self.dataframe_kwargs.append(kwargs)

    def download_button(self, **_kwargs):
        return None

    def subheader(self, _text: str):
        return None

    def write(self, _text: str):
        return None

    def rerun(self):
        return None


class DummyAgent:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(("run", kwargs))
        return AgentResult(
            tickers=["AAPL", "MSFT"],
            data_by_ticker={
                "AAPL": {
                    "history": pd.DataFrame({"Close": [1.0, 1.1]}),
                    "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
                },
                "MSFT": {
                    "history": pd.DataFrame({"Close": [1.0, 1.05]}),
                    "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
                },
            },
            weights={"AAPL": 0.5, "MSFT": 0.5},
            allocation={"AAPL": 500.0, "MSFT": 500.0},
            analysis_text="analysis",
            suggestions={},
        )

    def refine(self, **kwargs):
        self.calls.append(("refine", kwargs))
        return self.run(**kwargs)


class DummyAgentWithSuggestions(DummyAgent):
    def run(self, **kwargs):
        self.calls.append(("run", kwargs))
        return AgentResult(
            tickers=["AAPL", "MSFT"],
            data_by_ticker={
                "AAPL": {
                    "history": pd.DataFrame({"Close": [1.0, 1.1]}),
                    "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
                },
                "MSFT": {
                    "history": pd.DataFrame({"Close": [1.0, 1.05]}),
                    "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
                },
            },
            weights={"AAPL": 0.8, "MSFT": 0.2},
            allocation={"AAPL": 800.0, "MSFT": 200.0},
            analysis_text="analysis",
            suggestions={"add": [], "remove": [], "reweight": {"MSFT": 0.3, "AAPL": 0.7}},
            metadata={
                "reasoning_text": "Model reasoning trace",
                "tool_invocations": [
                    {"name": "generate_tickers", "arguments": {"tickers": ["AAPL", "MSFT"]}},
                    {"name": "fetch_ticker_data", "arguments": {"tickers": ["AAPL", "MSFT"]}},
                ],
            },
        )


class DummyAgentWithAllocationOnly(DummyAgent):
    def run(self, **kwargs):
        self.calls.append(("run", kwargs))
        return AgentResult(
            tickers=["AAPL", "MSFT"],
            data_by_ticker={
                "AAPL": {
                    "history": pd.DataFrame({"Close": [1.0, 1.1]}),
                    "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
                },
                "MSFT": {
                    "history": pd.DataFrame({"Close": [1.0, 1.05]}),
                    "financials": pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"]),
                },
            },
            weights={},
            allocation={"AAPL": 700.0, "MSFT": 300.0},
            analysis_text="analysis",
            suggestions={},
        )


class DummyAgentWithProgress(DummyAgent):
    def run(self, **kwargs):
        self.calls.append(("run", kwargs))

        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(0, 2, "AAPL")
            progress_callback(1, 2, "MSFT")

        status_callback = kwargs.get("status_callback")
        if status_callback is not None:
            status_callback("allocate_weights")
            status_callback("analyze_portfolio")

        return super().run(**kwargs)


class DummyClient:
    pass


def _base_config(api_key: str | None) -> dict:
    return {
        "app": {"title": "Title", "layout": "wide"},
        "ui": {
            "sidebar_header": "Header",
            "portfolio_size_label": "Portfolio Size ($)",
            "chat_placeholder": "Chat",
            "chat_intro": "Intro",
            "chat_tab_label": "Chat",
            "history_tab_label": "History",
            "portfolio_tab_label": "Portfolio",
            "starter_prompts_label": "Try one",
            "starter_prompts": ["Prompt A", "Prompt B"],
            "ticker_reply_template": "Suggested tickers: {tickers}",
            "post_analysis_nudge": "Check other tabs",
            "suggested_label": "Suggested",
            "download_prompt": "Download",
            "download_history_label": "History CSV",
            "history_empty_message": "History empty",
            "portfolio_empty_message": "Portfolio empty",
            "portfolio_output_label": "Recommended Portfolio",
            "portfolio_stats_label": "Portfolio Stats",
            "portfolio_returns_label": "Portfolio Returns",
            "portfolio_financials_label": "Portfolio Financials",
            "missing_api_key": "Missing",
            "missing_massive_key": "Missing Massive",
            "fetch_progress_start": "Fetching ticker data...",
            "fetch_progress_ticker": "Fetching {ticker} ({current}/{total})",
            "evaluator_accept_button": "Accept Changes",
            "evaluator_reject_button": "Keep Current Portfolio",
        },
        "dashboard": {
            "default_user_input": "default",
            "default_portfolio_size": 1000.0,
            "ticker_delimiter": ",",
        },
        "openrouter": {
            "api": {
                "api_key": api_key,
                "base_url": "https://example.com",
                "http_referer": "http://localhost",
                "app_title": "App",
            },
            "model_choices": ["anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash"],
        },
        "stocks": {
            "history_period": "1y",
            "financials_period": "quarterly",
            "max_tickers": 5,
            "financials_metrics": ["Total Revenue"],
        },
        "massive": {"api": {"api_key": "massive-key"}},
        "event_store": {"enabled": False, "schema_version": 2},
        "agent": {"model": "anthropic/claude-sonnet-4.5"},
    }


def test_run_dashboard_missing_openrouter_api_key(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt=None)
    monkeypatch.setattr("src.dashboard.st", st)

    run_dashboard(_base_config(api_key=None))

    assert sidebar.error_msg == "Missing"


def test_run_dashboard_missing_massive_api_key(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt=None)
    monkeypatch.setattr("src.dashboard.st", st)

    cfg = _base_config(api_key="ok")
    cfg["massive"]["api"].pop("api_key", None)
    run_dashboard(cfg)

    assert sidebar.error_msg == "Missing Massive"


def test_run_dashboard_renders_model_selector(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt=None)
    monkeypatch.setattr("src.dashboard.st", st)

    run_dashboard(_base_config(api_key="ok"))

    assert len(sidebar.selectbox_calls) == 1
    assert sidebar.selectbox_calls[0][0] == "Agent Model"


def test_run_dashboard_prompt_uses_single_agent(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt="build me a portfolio")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", DummyAgent)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "returns")

    run_dashboard(_base_config(api_key="ok"))

    assert sidebar.write_args == ("Suggested", ["AAPL", "MSFT"])
    assert any("Suggested tickers: AAPL, MSFT" in message for message in st.markdowns)
    assert any("analysis" in message for message in st.markdowns)
    assert st.progress_updates
    keys = [key for key, _label in st.button_calls]
    assert "accept_changes" in keys
    assert "reject_changes" in keys


def test_run_dashboard_shows_suggestions_and_action_buttons(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt="build me a portfolio")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", DummyAgentWithSuggestions)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "returns")

    run_dashboard(_base_config(api_key="ok"))

    assert any("Suggested portfolio changes" in message for message in st.markdowns)
    keys = [key for key, _label in st.button_calls]
    assert "accept_changes" in keys
    assert "reject_changes" in keys


def test_run_dashboard_shows_reasoning_without_tool_invocations(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt="build me a portfolio")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", DummyAgentWithSuggestions)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "returns")

    run_dashboard(_base_config(api_key="ok"))

    assert any("Reasoning" in message for message in st.markdowns)
    assert any("Model reasoning trace" in message for message in st.markdowns)
    assert not any("Tool invocations" in message for message in st.markdowns)
    assert not any("generate_tickers" in message for message in st.markdowns)


def test_run_dashboard_history_chart_uses_fetched_data(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt="build me a portfolio")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", DummyAgent)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *_args, **_kwargs: {})
    captured = {}

    def _plot_history(history_by_ticker, selected_tickers):
        captured["history_by_ticker"] = history_by_ticker
        captured["selected_tickers"] = selected_tickers
        return "history"

    monkeypatch.setattr("src.dashboard.plot_history", _plot_history)
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "returns")

    run_dashboard(_base_config(api_key="ok"))

    assert captured["selected_tickers"] == ["AAPL", "MSFT"]
    assert set(captured["history_by_ticker"].keys()) == {"AAPL", "MSFT"}
    assert any(kwargs.get("width") == "stretch" for kwargs in st.plot_kwargs)


def test_run_dashboard_preserves_non_equal_allocation_when_valid(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt="build me a portfolio")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", DummyAgentWithAllocationOnly)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "returns")

    captured = {}

    def _plot_allocation(allocation, title):
        captured["allocation"] = dict(allocation)
        return "allocation"

    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", _plot_allocation)

    run_dashboard(_base_config(api_key="ok"))

    assert captured["allocation"] == {"AAPL": 700.0, "MSFT": 300.0}


def test_run_dashboard_streams_assistant_llm_messages(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt="build me a portfolio")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", DummyAgent)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "returns")

    run_dashboard(_base_config(api_key="ok"))

    assert any("Suggested tickers: AAPL, MSFT" in message for message in st.streamed_messages)
    assert any("analysis" in message for message in st.streamed_messages)


def test_run_dashboard_shows_progress_for_weights_and_analysis(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, prompt="build me a portfolio")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", DummyAgentWithProgress)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "returns")

    run_dashboard(_base_config(api_key="ok"))

    progress_texts = [text or "" for _value, text in st.progress_updates]
    assert any("Computing portfolio weights" in text for text in progress_texts)
    assert any("Analyzing portfolio" in text for text in progress_texts)
