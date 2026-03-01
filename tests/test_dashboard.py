"""Unit tests for the dashboard module."""

import pandas as pd

from src.dashboard import (
    _extract_message_text,
    build_prompt,
    create_openrouter_client,
    fetch_stock_data,
    generate_tickers,
    limit_tickers,
    parse_tickers,
    run_dashboard,
)


class DummyAgg:
    """Minimal stub for a Massive.com Agg object."""

    def __init__(self, o, h, l, c, v, ts):  # noqa: E741
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.timestamp = ts


class DummyMetricField:
    """Stub for a financials metric field with a .value attribute."""

    def __init__(self, value):
        self.value = value

    def get(self, key, default=None):
        if key == "value":
            return self.value
        return default


class DummyIncomeStatement:
    def __init__(self):
        self.revenues = DummyMetricField(1_000_000)
        self.cost_of_revenue = DummyMetricField(500_000)
        self.operating_income_loss = DummyMetricField(300_000)
        self.net_income_loss = DummyMetricField(200_000)


class DummyCashFlowStatement:
    def __init__(self):
        self.depreciation_and_amortization = DummyMetricField(50_000)


class DummyFinancials:
    def __init__(self):
        self.income_statement = DummyIncomeStatement()
        self.cash_flow_statement = DummyCashFlowStatement()


class DummyStockFinancial:
    def __init__(self, report_date="2025-12-31"):
        self.period_of_report_date = report_date
        self.financials = DummyFinancials()


class DummyVx:
    """Stub for client.vx namespace."""

    def list_stock_financials(self, **_kwargs):
        return [DummyStockFinancial("2025-12-31"), DummyStockFinancial("2025-09-30")]


class DummyMassiveClient:
    """Minimal stub for massive.RESTClient."""

    def __init__(self):
        self.vx = DummyVx()

    def list_aggs(self, **_kwargs):
        return [
            DummyAgg(100, 105, 99, 102, 1000, 1700000000000),
            DummyAgg(102, 106, 101, 104, 1200, 1700086400000),
            DummyAgg(104, 108, 103, 107, 1100, 1700172800000),
        ]


class DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = DummyMessage(content)


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [DummyChoice(content)]


class DummyChatCompletions:
    def __init__(self, contents) -> None:
        self._contents = list(contents) if isinstance(contents, list) else [contents]

    def create(self, **_kwargs):
        if not self._contents:
            return DummyResponse("")
        return DummyResponse(self._contents.pop(0))


class DummyChat:
    def __init__(self, content) -> None:
        self.completions = DummyChatCompletions(content)


class DummyClient:
    """Minimal stub for OpenAI-compatible client."""

    def __init__(self, text) -> None:
        self.chat = DummyChat(text)


class DummySidebar:
    def __init__(self) -> None:
        self.session_state: dict | None = None
        self.header_text = None
        self.multiselect_args = None
        self.number_input_args = None
        self.error_msg = None
        self.write_args = None
        self.warning_msg = None

    def header(self, text: str) -> None:
        self.header_text = text

    def number_input(
        self,
        label: str,
        min_value: float,
        step: float,
        key: str | None = None,
    ) -> float:
        self.number_input_args = (label, min_value, step, key)
        if key and self.session_state is not None:
            self.session_state.setdefault(key, min_value)
            return self.session_state[key]
        return min_value

    def multiselect(self, label: str, options, default=None, key: str | None = None):
        self.multiselect_args = (label, options, default, key)
        selection = default or []
        if key and self.session_state is not None:
            if key not in self.session_state:
                self.session_state[key] = selection
            return self.session_state[key]
        return selection

    def write(self, label: str, value) -> None:
        self.write_args = (label, value)

    def error(self, msg: str) -> None:
        self.error_msg = msg

    def warning(self, msg: str) -> None:
        self.warning_msg = msg


class DummyContainer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyPlaceholder:
    def __init__(self, markdowns=None, progress_updates=None):
        self._markdowns = markdowns
        self._progress_updates = progress_updates

    def markdown(self, text: str) -> None:
        if self._markdowns is not None:
            self._markdowns.append(text)

    def progress(self, value: float, text: str | None = None):
        if self._progress_updates is not None:
            self._progress_updates.append((value, text))
        return self

    def empty(self) -> None:
        return None


class DummyStreamlit:
    def __init__(self, sidebar: DummySidebar, chat_input_value: str | None) -> None:
        self.sidebar = sidebar
        self.session_state = {}
        self.sidebar.session_state = self.session_state
        self._chat_input_value = chat_input_value
        self.title_text = None
        self.subheaders = []
        self.writes = []
        self.dataframes = []
        self.plots = []
        self.infos = []
        self.download_buttons = []
        self.captions = []
        self.chat_messages = []
        self.markdowns = []
        self.tabs_created = []
        self.warnings = []
        self.progress_updates = []
        self.placeholder_markdowns = []

    def title(self, text: str) -> None:
        self.title_text = text

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def write(self, text: str) -> None:
        self.writes.append(text)

    def dataframe(self, df: pd.DataFrame, **kwargs) -> None:
        self.dataframes.append(df)

    def pyplot(self, fig) -> None:
        self.plots.append(fig)

    def plotly_chart(self, fig, **_kwargs) -> None:
        self.plots.append(fig)

    def info(self, text: str) -> None:
        self.infos.append(text)

    def warning(self, text: str) -> None:
        self.warnings.append(text)

    def download_button(self, label: str, data, file_name: str, mime: str) -> None:
        self.download_buttons.append((label, file_name, mime, data))

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def columns(self, _spec, _gap: str | None = None):
        return [DummyContainer(), DummyContainer()]

    def tabs(self, labels):
        self.tabs_created = labels
        return [DummyContainer() for _ in labels]

    def progress(self, value: float, text: str | None = None):
        self.progress_updates.append((value, text))
        return DummyPlaceholder(progress_updates=self.progress_updates)

    def empty(self):
        return DummyPlaceholder(markdowns=self.placeholder_markdowns)

    def chat_input(self, _placeholder: str):
        return self._chat_input_value

    def chat_message(self, role: str):
        self.chat_messages.append(role)
        return DummyContainer()

    def markdown(self, text: str) -> None:
        self.markdowns.append(text)


def test_build_prompt():
    prompt = build_prompt("Hello {user_input}", "world")
    assert prompt == "Hello world"


def test_build_prompt_with_extra_kwargs():
    prompt = build_prompt("Pick {num_tickers} for {user_input}", "tech", num_tickers=10)
    assert prompt == "Pick 10 for tech"


def test_parse_tickers():
    tickers = parse_tickers("aapl, msft,  goog", delimiter=",")
    assert tickers == ["AAPL", "MSFT", "GOOG"]


def test_generate_tickers():
    client = DummyClient("AAPL, MSFT")
    tickers, raw = generate_tickers(
        client=client,
        prompt="test",
        system_prompt="system",
        model="model",
        max_tokens=10,
        temperature=0.1,
        delimiter=",")
    assert tickers == ["AAPL", "MSFT"]
    assert raw == "AAPL, MSFT"


def test_fetch_stock_data():
    client = DummyMassiveClient()

    data = fetch_stock_data("AAPL", history_period="1y", financials_period="quarterly", massive_client=client)

    assert "history" in data
    assert "financials" in data
    assert not data["history"].empty
    assert not data["financials"].empty


def test_limit_tickers():
    tickers = ["A", "B", "C"]
    assert limit_tickers(tickers, 2) == ["A", "B"]
    assert limit_tickers(tickers, 0) == []


def test_extract_message_text_dict_response():
    response = {"choices": [{"message": {"content": "AAPL, MSFT"}}]}
    assert _extract_message_text(response) == "AAPL, MSFT"


def test_create_openrouter_client(monkeypatch):
    class DummyOpenAI:
        def __init__(self, api_key: str, base_url: str, default_headers: dict):
            self.api_key = api_key
            self.base_url = base_url
            self.default_headers = default_headers

    monkeypatch.setattr("src.dashboard.OpenAI", DummyOpenAI)

    client = create_openrouter_client(
        api_key="key",
        base_url="https://example.com",
        headers={"X-Test": "1"},
    )

    assert client.api_key == "key"
    assert client.base_url == "https://example.com"
    assert client.default_headers == {"X-Test": "1"}


def _base_config(api_key: str | None) -> dict:
    return {
        "app": {"title": "Title", "layout": "wide"},
        "ui": {
            "sidebar_header": "Header",
            "portfolio_size_label": "Portfolio Size ($)",
            "chat_placeholder": "Chat",
            "chat_intro": "Intro",
            "chat_tab_label": "Chat",
            "ticker_reply_template": "Suggested tickers: {tickers}",
            "ticker_validation_error": "No valid tickers found.",
            "weights_fallback_message": "Could not parse weights. Using equal weights.",
            "weights_tickers_dropped": "Note: {dropped} received zero weight.",
            "fetch_progress_start": "Fetching ticker data...",
            "fetch_progress_ticker": "Fetching {ticker} ({current}/{total})",
            "history_fetch_warning": "No historical price data found for: {tickers}. These were skipped.",
            "history_fetch_all_failed": "Could not fetch historical price data for any suggested ticker. Please try a different request.",
            "history_empty_message": "History empty",
            "portfolio_empty_message": "Portfolio empty",
            "post_analysis_nudge": "Check other tabs",
            "portfolio_tab_link": "Open the **Portfolio** tab above to review allocation and performance.",
            "suggested_label": "Suggested",
            "ticker_header_template": "Data for {ticker}",
            "section_history": "History",
            "section_financials": "Financials",
            "download_prompt": "Download",
            "download_history_label": "History CSV",
            "download_financials_label": "Financials CSV",
            "missing_api_key": "Missing",
            "max_tickers_warning": "Limiting to {max_tickers}",
            "history_ticker_label": "History tickers",
            "history_tab_label": "History",
            "portfolio_tab_label": "Portfolio",
            "portfolio_output_label": "Recommended Portfolio",
            "portfolio_stats_label": "Portfolio Stats",
            "portfolio_returns_label": "Portfolio Returns",
            "portfolio_financials_label": "Portfolio Financials",
            "portfolio_stats_template": "Min {min} Max {max} Median {median} Current {current} Return {return_1y}",
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
            "models": {
                "ticker": "model",
                "weights": "model",
                "analysis": "model",
            },
            "outputs": {
                "ticker_max_tokens": 5,
                "weights_max_tokens": 5,
                "analysis_max_tokens": 500,
            },
            "temperatures": {
                "ticker": 0.1,
                "weights": 0.1,
                "analysis": 0.1,
            },
            "prompts": {
                "ticker_system": "system",
                "ticker_template": "Prompt {user_input}",
                "weights_system": "weights",
                "weights_template": "Weights {summary}",
                "analysis_system": "analysis",
                "analysis_template": "Summary {summary} Weights {weights}",
            },
        },
        "stocks": {
            "history_period": "1y",
            "financials_period": "quarterly",
            "max_tickers": 5,
            "financials_metrics": ["Total Revenue"],
        },
        "massive": {"api": {"api_key": "test-massive-key", "key_env_var": "MASSIVE_API_KEY"}},
    }


def test_run_dashboard_missing_api_key(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value=None)
    monkeypatch.setattr("src.dashboard.st", st)

    run_dashboard(_base_config(api_key=None))

    assert sidebar.error_msg == "Missing"


def test_run_dashboard_no_prompt(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value=None)
    monkeypatch.setattr("src.dashboard.st", st)

    def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("unexpected call")

    monkeypatch.setattr("src.dashboard.create_openrouter_client", should_not_be_called)

    run_dashboard(_base_config(api_key="key"))

    assert sidebar.error_msg is None


def test_run_dashboard_prompt_flow(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value="prompt")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr(
        "src.dashboard.create_openrouter_client",
        lambda **_kwargs: DummyClient(
            [
                "AAPL, MSFT",
                '{"weights": {"AAPL": 0.6, "MSFT": 0.4}}',
                "analysis",
            ]
        ),
    )
    monkeypatch.setattr(
        "src.dashboard.create_massive_client",
        lambda **_kwargs: DummyMassiveClient(),
    )
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "portfolio")
    monkeypatch.setattr(
        "src.dashboard.fetch_stock_data",
        lambda *_args, **_kwargs: {
            "history": pd.DataFrame({"Close": [1.0]}),
            "financials": pd.DataFrame({"metric": ["rev"]}),
        },
    )

    run_dashboard(_base_config(api_key="key"))

    assert sidebar.write_args == ("Suggested", ["AAPL", "MSFT"])
    assert len(st.tabs_created) == 3
    assert len(st.dataframes) == 1
    assert len(st.download_buttons) == 2
    assert len(st.plots) == 3
    assert st.progress_updates
    assert "Suggested tickers: AAPL, MSFT" in st.markdowns
    assert "analysis" in st.markdowns
    assert "Check other tabs" in st.markdowns
    assert "Open the **Portfolio** tab above to review allocation and performance." in st.markdowns


def test_run_dashboard_invalid_ticker_output(monkeypatch, caplog):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value="prompt")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr(
        "src.dashboard.create_openrouter_client",
        lambda **_kwargs: DummyClient(["??? , ###"]),
    )

    run_dashboard(_base_config(api_key="key"))

    assert "No valid tickers found." in st.markdowns
    assert "Raw model output from OpenRouter: ??? , ###" in caplog.text


def test_run_dashboard_weights_normalized(monkeypatch):
    """Weights that don't sum to 1.0 are normalized, not rejected."""
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value="prompt")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr(
        "src.dashboard.create_openrouter_client",
        lambda **_kwargs: DummyClient(
            [
                "AAPL, MSFT",
                '{"weights": {"AAPL": 0.9, "MSFT": 0.5}}',
                "analysis",
            ]
        ),
    )
    monkeypatch.setattr(
        "src.dashboard.create_massive_client",
        lambda **_kwargs: DummyMassiveClient(),
    )
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "portfolio")
    monkeypatch.setattr(
        "src.dashboard.fetch_stock_data",
        lambda *_args, **_kwargs: {
            "history": pd.DataFrame({"Close": [1.0]}),
            "financials": pd.DataFrame({"metric": ["rev"]}),
        },
    )

    run_dashboard(_base_config(api_key="key"))

    # Weights are normalized, not rejected — no sidebar warning
    assert sidebar.warning_msg is None
    # Analysis should still complete
    assert "analysis" in st.markdowns


def test_run_dashboard_weights_dropped_tickers(monkeypatch):
    """When LLM omits tickers from weights, user is informed."""
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value="prompt")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr(
        "src.dashboard.create_openrouter_client",
        lambda **_kwargs: DummyClient(
            [
                "AAPL, MSFT",
                '{"weights": {"AAPL": 1.0}}',
                "analysis",
            ]
        ),
    )
    monkeypatch.setattr(
        "src.dashboard.create_massive_client",
        lambda **_kwargs: DummyMassiveClient(),
    )
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "portfolio")
    monkeypatch.setattr(
        "src.dashboard.fetch_stock_data",
        lambda *_args, **_kwargs: {
            "history": pd.DataFrame({"Close": [1.0]}),
            "financials": pd.DataFrame({"metric": ["rev"]}),
        },
    )

    run_dashboard(_base_config(api_key="key"))

    assert any("MSFT" in msg and "zero weight" in msg for msg in st.markdowns)
    assert "analysis" in st.markdowns


def test_run_dashboard_unparseable_weights_fallback(monkeypatch):
    """When weight parsing fails completely, equal weights are used."""
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value="prompt")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr(
        "src.dashboard.create_openrouter_client",
        lambda **_kwargs: DummyClient(
            [
                "AAPL, MSFT",
                "not valid json at all",
                "analysis",
            ]
        ),
    )
    monkeypatch.setattr(
        "src.dashboard.create_massive_client",
        lambda **_kwargs: DummyMassiveClient(),
    )
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "portfolio")
    monkeypatch.setattr(
        "src.dashboard.fetch_stock_data",
        lambda *_args, **_kwargs: {
            "history": pd.DataFrame({"Close": [1.0]}),
            "financials": pd.DataFrame({"metric": ["rev"]}),
        },
    )

    run_dashboard(_base_config(api_key="key"))

    assert "Could not parse weights. Using equal weights." in st.markdowns
    assert "analysis" in st.markdowns


def test_run_dashboard_tabs_show_empty_states_without_tickers(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value=None)
    monkeypatch.setattr("src.dashboard.st", st)

    run_dashboard(_base_config(api_key="key"))

    assert "History empty" in st.infos
    assert "Portfolio empty" in st.infos


def test_run_dashboard_warns_for_missing_history(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value="prompt")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr(
        "src.dashboard.create_openrouter_client",
        lambda **_kwargs: DummyClient(
            [
                "AAPL, MSFT",
                '{"weights": {"AAPL": 1.0}}',
                "analysis",
            ]
        ),
    )
    monkeypatch.setattr(
        "src.dashboard.create_massive_client",
        lambda **_kwargs: DummyMassiveClient(),
    )
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_portfolio_allocation", lambda *_args, **_kwargs: "allocation")
    monkeypatch.setattr("src.dashboard.plot_portfolio_returns", lambda *_args, **_kwargs: "portfolio")

    def _fetch(*args, **_kwargs):
        ticker = args[0]
        if ticker == "MSFT":
            return {
                "history": pd.DataFrame(),
                "financials": pd.DataFrame({"metric": ["rev"]}),
            }
        return {
            "history": pd.DataFrame({"Close": [1.0]}),
            "financials": pd.DataFrame({"metric": ["rev"]}),
        }

    monkeypatch.setattr("src.dashboard.fetch_stock_data", _fetch)

    run_dashboard(_base_config(api_key="key"))

    assert any("MSFT" in warning for warning in st.warnings)
    assert any("MSFT" in message for message in st.markdowns)
    assert sidebar.write_args == ("Suggested", ["AAPL"])


def test_run_dashboard_all_history_fetches_fail(monkeypatch):
    sidebar = DummySidebar()
    st = DummyStreamlit(sidebar, chat_input_value="prompt")
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr(
        "src.dashboard.create_openrouter_client",
        lambda **_kwargs: DummyClient(["AAPL, MSFT"]),
    )
    monkeypatch.setattr(
        "src.dashboard.create_massive_client",
        lambda **_kwargs: DummyMassiveClient(),
    )
    monkeypatch.setattr(
        "src.dashboard.fetch_stock_data",
        lambda *_args, **_kwargs: {
            "history": pd.DataFrame(),
            "financials": pd.DataFrame(),
        },
    )

    run_dashboard(_base_config(api_key="key"))

    assert "Could not fetch historical price data for any suggested ticker. Please try a different request." in st.markdowns
    assert sidebar.write_args is None