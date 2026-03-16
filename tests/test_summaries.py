"""Unit tests for data summarization helpers."""

import pandas as pd

from src.summaries import (
    build_portfolio_returns_series,
    summarize_financials_latest,
    summarize_history_stats,
    summarize_portfolio_financials,
    summarize_portfolio_stats,
    build_ticker_summary,
)


def test_summarize_history_stats():
    history = pd.DataFrame({"Close": [10.0, 12.0, 11.0, 9.0]})

    stats = summarize_history_stats(history)

    assert stats["min"] == 9.0
    assert stats["max"] == 12.0
    assert stats["median"] == 10.5
    assert stats["current"] == 9.0


def test_summarize_financials_latest():
    financials = pd.DataFrame(
        {"2024-01-01": [100, 50], "2024-04-01": [110, 60]},
        index=["Total Revenue", "EBITDA"],
    )

    summary = summarize_financials_latest(financials, ["Total Revenue", "EBITDA"])

    assert summary["Total Revenue"] == 110
    assert summary["EBITDA"] == 60


def test_build_ticker_summary():
    data = {
        "history": pd.DataFrame({"Close": [10.0, 12.0, 11.0, 9.0]}),
        "financials": pd.DataFrame(
            {"2024-01-01": [100], "2024-04-01": [110]},
            index=["Total Revenue"],
        ),
    }

    summary_text = build_ticker_summary(
        ticker="AAPL",
        data=data,
    )

    assert '"t":"AAPL"' in summary_text
    assert '"p":' in summary_text


def test_build_portfolio_returns_series():
    history = {
        "AAPL": pd.DataFrame({"Close": [10.0, 11.0, 12.0]}),
        "MSFT": pd.DataFrame({"Close": [20.0, 22.0, 21.0]}),
    }

    series = build_portfolio_returns_series(history, {"AAPL": 0.5, "MSFT": 0.5})

    assert not series.empty
    assert series.name == "Portfolio"


def test_summarize_portfolio_stats():
    series = pd.Series([1.0, 1.05, 1.02])

    stats = summarize_portfolio_stats(series)

    assert stats["min"] == 1.0
    assert stats["current"] == 1.02
    assert round(stats["return_1y"], 2) == 0.02


def test_summarize_portfolio_financials():
    financials = {
        "AAPL": pd.DataFrame(
            {"2024-01-01": [100], "2024-04-01": [110]},
            index=["Total Revenue"],
        ),
        "MSFT": pd.DataFrame(
            {"2024-01-01": [200], "2024-04-01": [220]},
            index=["Total Revenue"],
        ),
    }

    summary = summarize_portfolio_financials(
        financials,
        weights={"AAPL": 0.5, "MSFT": 0.5},
        metrics=["Total Revenue"],
    )

    assert summary["Total Revenue"] == 165.0
