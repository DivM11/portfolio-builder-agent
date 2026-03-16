"""Unit tests for plot helpers."""

import pandas as pd

from src.plots import plot_financials, plot_history, plot_portfolio_allocation


def _history_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0, 101.5, 99.8]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )


def test_plot_history_selected_tickers():
    history = {"AAPL": _history_df(), "MSFT": _history_df()}
    fig = plot_history(history, selected_tickers=["AAPL"])

    assert fig is not None
    assert len(fig.data) == 1


def test_plot_history_accepts_lowercase_close_column():
    history = {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 101.5, 99.8]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
    }

    fig = plot_history(history)

    assert fig is not None
    assert len(fig.data) == 1


def test_plot_history_returns_none_for_all_nan_close_values():
    history = {
        "AAPL": pd.DataFrame(
            {"Close": [None, float("nan"), None]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
    }

    fig = plot_history(history)

    assert fig is None


def test_plot_portfolio_allocation():
    allocation = {"AAPL": 600.0, "MSFT": 400.0}
    fig = plot_portfolio_allocation(allocation, title="Test Allocation")

    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].orientation is None  # vertical (default)


def test_plot_portfolio_allocation_empty():
    fig = plot_portfolio_allocation({})

    assert fig is None
