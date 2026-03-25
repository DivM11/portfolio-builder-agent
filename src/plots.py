"""Plotting utilities for the dashboard."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure


def _apply_gridlines(fig: Figure) -> None:
    fig.update_xaxes(showgrid=True, gridcolor="#E6E6E6")
    fig.update_yaxes(showgrid=True, gridcolor="#E6E6E6")


def plot_history(
    history_by_ticker: dict[str, pd.DataFrame],
    selected_tickers: Iterable[str] | None = None,
) -> Figure | None:
    """Plot closing price history for each ticker on a single chart."""
    if not history_by_ticker:
        return None

    tickers = list(selected_tickers) if selected_tickers is not None else list(history_by_ticker)
    fig = go.Figure()
    for ticker in tickers:
        history = history_by_ticker.get(ticker)
        if history is None or history.empty:
            continue

        close_col = None
        for candidate in history.columns:
            if str(candidate).strip().lower() == "close":
                close_col = candidate
                break
        if close_col is None:
            continue

        close_series = pd.to_numeric(history[close_col], errors="coerce").dropna()
        if close_series.empty:
            continue

        x_values = close_series.index
        if isinstance(x_values, pd.DatetimeIndex):
            x_values = x_values.sort_values()
            close_series = close_series.reindex(x_values)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=close_series.values,
                mode="lines",
                name=ticker,
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        title="Historical Price (Close) vs Date",
        xaxis_title="Date",
        yaxis_title="Close Price",
        legend_title="Ticker",
    )
    _apply_gridlines(fig)
    return fig


def plot_portfolio_returns(portfolio_series: pd.Series, title: str) -> Figure | None:
    if portfolio_series is None or portfolio_series.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_series.index,
            y=portfolio_series.values,
            mode="lines",
            name="Portfolio",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
    )
    _apply_gridlines(fig)
    return fig


def plot_portfolio_comparison(
    series_by_portfolio: dict[str, pd.Series],
    title: str = "Portfolio Comparison",
) -> Figure | None:
    """Overlay normalized returns for multiple portfolios on a single chart."""
    if not series_by_portfolio:
        return None

    fig = go.Figure()
    for name, series in series_by_portfolio.items():
        if series is None or series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend_title="Portfolio",
    )
    _apply_gridlines(fig)
    return fig


def plot_portfolio_allocation(
    allocation: dict[str, float],
    title: str = "Recommended Portfolio",
) -> Figure | None:
    """Vertical bar chart showing dollar allocation per ticker with annotations."""
    if not allocation:
        return None

    tickers = list(allocation.keys())
    amounts = [allocation[t] for t in tickers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=amounts,
            text=[f"${a:,.0f}" for a in amounts],
            textposition="outside",
            marker_color="#636EFA",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Allocation ($)",
        xaxis={"categoryorder": "total descending"},
    )
    _apply_gridlines(fig)
    return fig
