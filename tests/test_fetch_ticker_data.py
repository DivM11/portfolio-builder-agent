"""Unit tests for the fetch_ticker_data tool."""

from __future__ import annotations

from src.tickr_data_manager import TickrDataFetchResult
from src.tools.fetch_ticker_data import fetch_ticker_data_tool, tool_definition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyDataManager:
    def __init__(self, result: TickrDataFetchResult) -> None:
        self._result = result
        self.last_kwargs: dict[str, object] = {}

    def fetch_for_tickers(self, **kwargs) -> TickrDataFetchResult:
        self.last_kwargs = kwargs
        return self._result


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


def test_tool_definition_has_expected_name() -> None:
    definition = tool_definition()

    assert definition["type"] == "function"
    assert definition["function"]["name"] == "fetch_ticker_data"


# ---------------------------------------------------------------------------
# Tool behavior
# ---------------------------------------------------------------------------


def test_fetch_ticker_data_tool_reports_fetched_cached_failed() -> None:
    manager = DummyDataManager(
        TickrDataFetchResult(
            data_by_ticker={},
            tickers_with_history=["AAPL", "MSFT"],
            failed_history_by_status={
                "rate_limited": ["NVDA"],
                "not_found": [],
                "empty_data": ["TSLA"],
                "unexpected_error": [],
            },
            fetched_tickers=["MSFT"],
        )
    )

    payload = fetch_ticker_data_tool(
        {"tickers": ["aapl", "msft", "nvda", "tsla"]},
        tickr_data_manager=manager,
        stock_data_fetcher=lambda **_kwargs: {},
        history_period="1y",
        massive_client=object(),
    )

    assert payload["fetched"] == ["MSFT"]
    assert payload["cached"] == ["AAPL"]
    assert payload["available_tickers"] == ["AAPL", "MSFT"]
    assert payload["failed"] == {"NVDA": "rate_limited", "TSLA": "empty_data"}


def test_fetch_ticker_data_tool_passes_progress_callback() -> None:
    manager = DummyDataManager(
        TickrDataFetchResult(
            data_by_ticker={},
            tickers_with_history=[],
            failed_history_by_status={
                "rate_limited": [],
                "not_found": [],
                "empty_data": [],
                "unexpected_error": [],
            },
            fetched_tickers=[],
        )
    )

    callback_calls: list[tuple[int, int, str]] = []

    def _progress(current: int, total: int, ticker: str) -> None:
        callback_calls.append((current, total, ticker))

    fetch_ticker_data_tool(
        {"tickers": ["AAPL"]},
        tickr_data_manager=manager,
        stock_data_fetcher=lambda **_kwargs: {},
        history_period="6mo",
        massive_client=object(),
        progress_callback=_progress,
    )

    assert manager.last_kwargs["history_period"] == "6mo"
    assert callable(manager.last_kwargs["progress_callback"])
    # The callback is forwarded to manager; tool itself does not invoke it.
    assert callback_calls == []
