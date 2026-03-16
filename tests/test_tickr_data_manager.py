"""Unit tests for ticker data manager."""

import pandas as pd

from src.tickr_data_manager import TickrDataManager


def _payload(close_values, status="ok"):
    history = pd.DataFrame({"Close": close_values}) if close_values else pd.DataFrame()
    return {"history": history, "financials": pd.DataFrame(), "history_status": status}


def test_tickr_data_manager_caches_and_fetches_missing_only():
    manager = TickrDataManager()
    calls = []

    def fetcher(**kwargs):
        calls.append(kwargs["ticker"])
        if kwargs["ticker"] == "AAPL":
            return _payload([100.0, 101.0])
        return _payload([200.0, 201.0])

    first = manager.fetch_for_tickers(
        tickers=["AAPL"],
        fetcher=fetcher,
        history_period="1y",
        massive_client=object(),
    )
    second = manager.fetch_for_tickers(
        tickers=["AAPL", "MSFT"],
        fetcher=fetcher,
        history_period="1y",
        massive_client=object(),
    )

    assert first.tickers_with_history == ["AAPL"]
    assert second.tickers_with_history == ["AAPL", "MSFT"]
    assert calls == ["AAPL", "MSFT"]
    assert "AAPL" in manager.cache


def test_tickr_data_manager_tracks_rate_limited_failures():
    manager = TickrDataManager()

    def fetcher(**kwargs):
        _ = kwargs
        return _payload([], status="rate_limited")

    result = manager.fetch_for_tickers(
        tickers=["AAPL"],
        fetcher=fetcher,
        history_period="1y",
        massive_client=object(),
    )

    assert result.tickers_with_history == []
    assert result.failed_history_by_status["rate_limited"] == ["AAPL"]


def test_tickr_data_manager_passes_client_keyword_to_fetcher():
    manager = TickrDataManager()
    seen = {}

    def fetcher(*, client, ticker, history_period):
        seen["client"] = client
        seen["ticker"] = ticker
        seen["history_period"] = history_period
        return _payload([100.0, 101.0])

    token_client = object()
    result = manager.fetch_for_tickers(
        tickers=["AAPL"],
        fetcher=fetcher,
        history_period="1y",
        massive_client=token_client,
    )

    assert result.tickers_with_history == ["AAPL"]
    assert seen == {
        "client": token_client,
        "ticker": "AAPL",
        "history_period": "1y",
    }
