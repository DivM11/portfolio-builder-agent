"""Tests for multi-portfolio session management."""

from __future__ import annotations

import pandas as pd
import pytest

from src.agent_models import AgentResult, PortfolioState
from src.dashboard import _restore_portfolio, _save_current_portfolio, run_dashboard
from src.plots import plot_portfolio_comparison


# ---------------------------------------------------------------------------
# PortfolioState unit tests
# ---------------------------------------------------------------------------


def test_portfolio_state_requires_name() -> None:
    with pytest.raises(TypeError):
        PortfolioState()  # type: ignore[call-arg]


def test_portfolio_state_defaults() -> None:
    ps = PortfolioState(name="Test")
    assert ps.name == "Test"
    assert ps.tickers == []
    assert ps.weights == {}
    assert ps.portfolio_size == 1000.0
    assert ps.is_processing is False
    assert ps.pending_prompt is None
    assert ps.awaiting_user_decision is False
    assert ps.latest_result is None
    assert ps.portfolio_series is None
    assert ps.allocation_table_df is None
    assert ps.portfolio_agent is None


# ---------------------------------------------------------------------------
# plot_portfolio_comparison
# ---------------------------------------------------------------------------


def _make_series(start: float = 1.0, length: int = 5) -> pd.Series:
    return pd.Series(
        [start + i * 0.01 for i in range(length)],
        index=pd.date_range("2024-01-01", periods=length),
    )


def test_plot_portfolio_comparison_returns_figure() -> None:
    series_map = {"Portfolio 1": _make_series(1.0), "Portfolio 2": _make_series(0.95)}
    fig = plot_portfolio_comparison(series_map)
    assert fig is not None
    assert len(fig.data) == 2
    names = {trace.name for trace in fig.data}
    assert names == {"Portfolio 1", "Portfolio 2"}


def test_plot_portfolio_comparison_returns_none_when_empty() -> None:
    assert plot_portfolio_comparison({}) is None


def test_plot_portfolio_comparison_skips_empty_series() -> None:
    series_map = {
        "Full": _make_series(),
        "Empty": pd.Series(dtype=float),
    }
    fig = plot_portfolio_comparison(series_map)
    # Only 1 non-empty series → not enough to compare → figure has 1 trace but still returns
    assert fig is not None
    names = [trace.name for trace in fig.data]
    assert "Full" in names
    assert "Empty" not in names


def test_plot_portfolio_comparison_all_empty_returns_none() -> None:
    series_map = {"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)}
    assert plot_portfolio_comparison(series_map) is None


# ---------------------------------------------------------------------------
# _save_current_portfolio / _restore_portfolio helpers
# (tested via monkeypatching st.session_state)
# ---------------------------------------------------------------------------


class _FakeSt:
    """Minimal Streamlit stub for save/restore helper tests."""

    def __init__(self, initial: dict | None = None) -> None:
        self.session_state: dict = dict(initial or {})


def test_save_current_portfolio_copies_flat_state(monkeypatch) -> None:
    fake_st = _FakeSt(
        {
            "tickers": ["AAPL", "MSFT"],
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
            "portfolio_size": 2000.0,
            "messages": [{"role": "assistant", "content": "Hi"}],
            "analysis_text": "Good portfolio",
        }
    )
    monkeypatch.setattr("src.dashboard.st", fake_st)

    ps = PortfolioState(name="My Portfolio")
    _save_current_portfolio(ps)

    assert ps.tickers == ["AAPL", "MSFT"]
    assert ps.weights == {"AAPL": 0.6, "MSFT": 0.4}
    assert ps.portfolio_size == 2000.0
    assert ps.analysis_text == "Good portfolio"


def test_restore_portfolio_writes_flat_state(monkeypatch) -> None:
    fake_st = _FakeSt({"portfolio_size": 1000.0})
    monkeypatch.setattr("src.dashboard.st", fake_st)

    ps = PortfolioState(
        name="Saved",
        tickers=["GOOG"],
        weights={"GOOG": 1.0},
        portfolio_size=3000.0,
        analysis_text="Saved analysis",
    )
    _restore_portfolio(ps, chat_intro="Welcome")

    ss = fake_st.session_state
    assert ss["tickers"] == ["GOOG"]
    assert ss["weights"] == {"GOOG": 1.0}
    assert ss["portfolio_size"] == 3000.0
    assert ss["analysis_text"] == "Saved analysis"


def test_restore_portfolio_sets_typed_defaults_when_none(monkeypatch) -> None:
    fake_st = _FakeSt({})
    monkeypatch.setattr("src.dashboard.st", fake_st)

    ps = PortfolioState(name="Empty")  # all defaults → None for complex types
    _restore_portfolio(ps, chat_intro="Hello")

    ss = fake_st.session_state
    assert isinstance(ss["portfolio_series"], pd.Series)
    assert isinstance(ss["allocation_table_df"], pd.DataFrame)
    assert isinstance(ss["latest_result"], AgentResult)
    assert isinstance(ss["tickers"], list)
    # When messages is empty, chat_intro is injected
    assert any("Hello" in m.get("content", "") for m in ss["messages"])


# ---------------------------------------------------------------------------
# Dashboard integration tests
# ---------------------------------------------------------------------------


class _DummyContainer:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _DummyPlaceholder:
    def progress(self, *_args, **_kwargs):
        return self

    def empty(self):
        pass


class _DummySidebar:
    def __init__(self, *, new_portfolio_clicked: bool = False, delete_clicked: bool = False):
        self.session_state: dict = {}
        self.error_msg: str | None = None
        self.selectbox_calls: list = []
        self.button_calls: list = []
        self._new_portfolio_clicked = new_portfolio_clicked
        self._delete_clicked = delete_clicked
        self.write_args = None

    def header(self, _text):
        pass

    def number_input(self, _label, min_value=0.0, step=100.0, key=None):
        if key:
            self.session_state.setdefault(key, min_value)
            return self.session_state[key]
        return min_value

    def selectbox(self, label, options, index=0, key=None):
        self.selectbox_calls.append((label, options, index, key))
        chosen = options[index] if options else None
        if key:
            self.session_state.setdefault(key, chosen)
            return self.session_state[key]
        return chosen

    def error(self, msg):
        self.error_msg = msg

    def button(self, label, key=None):
        key_to_use = key or label
        self.button_calls.append((key_to_use, label))
        if key_to_use == "new_portfolio_btn":
            return self._new_portfolio_clicked
        if key_to_use == "delete_portfolio_btn":
            return self._delete_clicked
        return False

    def write(self, label, value):
        self.write_args = (label, value)


class _DummySt:
    def __init__(self, sidebar: _DummySidebar, prompt: str | None = None):
        self.sidebar = sidebar
        self.session_state: dict = {}
        self.sidebar.session_state = self.session_state
        self._prompt = prompt
        self.markdowns: list[str] = []
        self.writes: list[str] = []
        self.subheaders: list[str] = []
        self.plot_kwargs: list[dict] = []
        self.dataframe_inputs: list = []
        self.dataframe_kwargs: list[dict] = []
        self.button_calls: list[tuple[str, str]] = []
        self.expander_calls: list = []
        self.infos: list[str] = []
        self.progress_updates: list = []
        self.rerun_count = 0
        self.streamed_messages: list[str] = []
        self.captions: list[str] = []

    def title(self, _text):
        pass

    def tabs(self, labels):
        return [_DummyContainer() for _ in labels]

    def caption(self, text):
        self.captions.append(text)

    def columns(self, count):
        return [_DummyContainer() for _ in range(count)]

    def button(self, label, key=None):
        key_to_use = key or label
        self.button_calls.append((key_to_use, label))
        return False

    def chat_input(self, _placeholder):
        return self._prompt

    def chat_message(self, _role, avatar=None):
        return _DummyContainer()

    def markdown(self, text):
        self.markdowns.append(text)

    def write_stream(self, stream):
        chunks = list(stream)
        text = "".join(str(c) for c in chunks)
        self.streamed_messages.append(text)
        self.markdowns.append(text)
        return text

    def progress(self, value, text=None):
        self.progress_updates.append((value, text))
        return _DummyPlaceholder()

    def info(self, text):
        self.infos.append(text)

    def plotly_chart(self, _fig, **kwargs):
        self.plot_kwargs.append(kwargs)

    def dataframe(self, df, **kwargs):
        self.dataframe_inputs.append(df)
        self.dataframe_kwargs.append(kwargs)

    def download_button(self, **_kwargs):
        pass

    def subheader(self, text):
        self.subheaders.append(text)

    def write(self, text):
        self.writes.append(str(text))

    def expander(self, label, expanded=False):
        self.expander_calls.append((label, expanded))
        return _DummyContainer()

    def rerun(self):
        self.rerun_count += 1


def _base_config() -> dict:
    return {
        "app": {"title": "Title", "layout": "wide"},
        "ui": {
            "sidebar_header": "Preferences",
            "portfolio_size_label": "Portfolio Size ($)",
            "chat_placeholder": "Chat",
            "chat_intro": "Intro",
            "chat_tab_label": "Chat",
            "history_tab_label": "History",
            "portfolio_tab_label": "Portfolio",
            "starter_prompts_label": "Try one",
            "starter_prompts": [],
            "ticker_reply_template": "Suggested tickers: {tickers}",
            "post_analysis_nudge": "Check other tabs",
            "suggested_label": "Suggested",
            "download_prompt": "Download",
            "download_history_label": "History CSV",
            "history_empty_message": "History empty",
            "portfolio_empty_message": "Portfolio empty",
            "portfolio_output_label": "Portfolio",
            "portfolio_stats_label": "Stats",
            "portfolio_returns_label": "Returns",
            "missing_api_key": "Missing key",
            "missing_massive_key": "Missing Massive",
            "fetch_progress_start": "Fetching...",
            "fetch_progress_ticker": "Fetching {ticker} ({current}/{total})",
            "evaluator_accept_button": "Accept",
            "evaluator_reject_button": "Keep",
        },
        "dashboard": {
            "default_user_input": "default",
            "default_portfolio_size": 1000.0,
            "ticker_delimiter": ",",
            "max_portfolios": 3,
        },
        "openrouter": {
            "api": {
                "api_key": "openrouter-key",
                "base_url": "https://example.com",
                "http_referer": "http://localhost",
                "app_title": "App",
            },
            "model_choices": [],
        },
        "stocks": {"history_period": "1y", "max_tickers": 5},
        "massive": {"api": {"api_key": "massive-key"}},
        "event_store": {"enabled": False, "schema_version": 2},
        "agent": {"model": "gpt-4"},
    }


class _DummyAgent:
    def __init__(self, *_, **__):
        self.calls: list = []

    def run(self, **kwargs):
        self.calls.append(("run", kwargs))
        return AgentResult(
            tickers=["AAPL"],
            data_by_ticker={"AAPL": {"history": pd.DataFrame({"Close": [1.0, 1.1]})}},
            weights={"AAPL": 1.0},
            allocation={"AAPL": 1000.0},
            analysis_text="ok",
            suggestions={},
        )

    def refine(self, **kwargs):
        return self.run(**kwargs)


def _patch_common(monkeypatch, st_mock):
    monkeypatch.setattr("src.dashboard.st", st_mock)
    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_: object())
    monkeypatch.setattr("src.dashboard.LLMService", lambda *a, **k: object())
    monkeypatch.setattr("src.dashboard.PortfolioAgent", _DummyAgent)
    monkeypatch.setattr("src.dashboard.create_massive_client", lambda *a, **k: object())
    monkeypatch.setattr("src.dashboard.fetch_stock_data", lambda *a, **k: {})
    monkeypatch.setattr("src.dashboard.plot_history", lambda *a, **k: None)
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *a, **k: None)
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *a, **k: None)
    monkeypatch.setattr("src.dashboard.plot_portfolio_comparison", lambda *a, **k: None)


def test_portfolio_selector_shown_in_sidebar(monkeypatch) -> None:
    sidebar = _DummySidebar()
    st_mock = _DummySt(sidebar)
    _patch_common(monkeypatch, st_mock)

    run_dashboard(_base_config())

    selector_labels = [call[0] for call in sidebar.selectbox_calls]
    assert "Portfolio" in selector_labels


def test_new_portfolio_button_shown_when_below_limit(monkeypatch) -> None:
    sidebar = _DummySidebar()
    st_mock = _DummySt(sidebar)
    _patch_common(monkeypatch, st_mock)

    run_dashboard(_base_config())

    btn_keys = [key for key, _label in sidebar.button_calls]
    assert "new_portfolio_btn" in btn_keys


def test_delete_button_hidden_when_only_one_portfolio(monkeypatch) -> None:
    sidebar = _DummySidebar()
    st_mock = _DummySt(sidebar)
    _patch_common(monkeypatch, st_mock)

    run_dashboard(_base_config())

    btn_keys = [key for key, _label in sidebar.button_calls]
    assert "delete_portfolio_btn" not in btn_keys


def test_new_portfolio_button_creates_portfolio_and_reruns(monkeypatch) -> None:
    sidebar = _DummySidebar(new_portfolio_clicked=True)
    st_mock = _DummySt(sidebar)
    _patch_common(monkeypatch, st_mock)

    run_dashboard(_base_config())

    # rerun() should have been called due to new portfolio creation
    assert st_mock.rerun_count >= 1
    # portfolios dict should now have 2 entries
    portfolios = st_mock.session_state.get("portfolios", {})
    assert len(portfolios) == 2


def test_new_portfolio_button_blocked_at_max_portfolios(monkeypatch) -> None:
    sidebar = _DummySidebar(new_portfolio_clicked=True)
    st_mock = _DummySt(sidebar)
    _patch_common(monkeypatch, st_mock)

    cfg = _base_config()
    cfg["dashboard"]["max_portfolios"] = 1  # already at max with 1 portfolio

    run_dashboard(cfg)

    # "new_portfolio_btn" button should not even be rendered
    btn_keys = [key for key, _label in sidebar.button_calls]
    assert "new_portfolio_btn" not in btn_keys
    portfolios = st_mock.session_state.get("portfolios", {})
    assert len(portfolios) == 1


def test_delete_portfolio_shown_with_multiple_portfolios(monkeypatch) -> None:
    """When two portfolios exist, the delete button should be visible."""
    from src.agent_models import PortfolioState

    sidebar = _DummySidebar()
    st_mock = _DummySt(sidebar)
    _patch_common(monkeypatch, st_mock)

    # Pre-populate two portfolios in session state
    two_portfolios = {
        "p1": PortfolioState(name="Portfolio 1"),
        "p2": PortfolioState(name="Portfolio 2"),
    }
    st_mock.session_state["portfolios"] = two_portfolios
    st_mock.session_state["current_portfolio_id"] = "p1"

    run_dashboard(_base_config())

    btn_keys = [key for key, _label in sidebar.button_calls]
    assert "delete_portfolio_btn" in btn_keys


def test_delete_portfolio_switches_to_remaining(monkeypatch) -> None:
    from src.agent_models import PortfolioState

    sidebar = _DummySidebar(delete_clicked=True)
    st_mock = _DummySt(sidebar)
    _patch_common(monkeypatch, st_mock)

    two_portfolios = {
        "p1": PortfolioState(name="Portfolio 1"),
        "p2": PortfolioState(name="Portfolio 2"),
    }
    st_mock.session_state["portfolios"] = two_portfolios
    st_mock.session_state["current_portfolio_id"] = "p1"

    run_dashboard(_base_config())

    assert st_mock.rerun_count >= 1
    remaining_portfolios = st_mock.session_state.get("portfolios", {})
    assert "p1" not in remaining_portfolios
    assert "p2" in remaining_portfolios
    assert st_mock.session_state.get("current_portfolio_id") == "p2"
