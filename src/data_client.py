"""Data client for fetching stock data from Massive.com (formerly Polygon.io)."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from massive import RESTClient

logger = logging.getLogger(__name__)

HISTORY_STATUS_OK = "ok"
HISTORY_STATUS_RATE_LIMITED = "rate_limited"
HISTORY_STATUS_NOT_FOUND = "not_found"
HISTORY_STATUS_EMPTY_DATA = "empty_data"
HISTORY_STATUS_UNEXPECTED_ERROR = "unexpected_error"

# portfolio-builder-style period strings → number of calendar days
_PERIOD_DAYS: dict[str, int] = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}

# Mapping: portfolio-builder metric name → Massive.com income_statement attribute
_INCOME_METRIC_MAP: dict[str, str] = {
    "Total Revenue": "revenues",
    "Cost Of Revenue": "cost_of_revenue",
    "Operating Income": "operating_income_loss",
    "Net Income": "net_income_loss",
}


def create_massive_client(api_key: str) -> RESTClient:
    """Create an authenticated Massive.com REST client."""
    return RESTClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------


def _classify_history_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return HISTORY_STATUS_RATE_LIMITED
    if "404" in text or "not found" in text or "unknown ticker" in text:
        return HISTORY_STATUS_NOT_FOUND
    return HISTORY_STATUS_UNEXPECTED_ERROR


def fetch_price_history_with_status(
    client: RESTClient,
    ticker: str,
    period: str = "1y",
) -> tuple[pd.DataFrame, str]:
    """Fetch daily OHLCV price history plus a normalized fetch status."""
    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    days = _PERIOD_DAYS.get(period, 365)
    to_date = date.today()
    from_date = to_date - timedelta(days=days)

    try:
        aggs: list[Any] = list(
            client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=from_date.isoformat(),
                to=to_date.isoformat(),
                adjusted=True,
                sort="asc",
                limit=50000,
            )
        )
    except Exception as exc:
        status = _classify_history_exception(exc)
        logger.exception("Massive.com list_aggs failed for %s status=%s", ticker, status)
        return empty_df, status

    if not aggs:
        return empty_df, HISTORY_STATUS_EMPTY_DATA

    df = pd.DataFrame(
        [
            {
                "Open": a.open,
                "High": a.high,
                "Low": a.low,
                "Close": a.close,
                "Volume": a.volume,
            }
            for a in aggs
        ],
        index=pd.to_datetime([a.timestamp for a in aggs], unit="ms"),
    )
    df.index.name = "Date"
    return df, HISTORY_STATUS_OK


def fetch_price_history(
    client: RESTClient,
    ticker: str,
    period: str = "1y",
) -> pd.DataFrame:
    """Fetch daily OHLCV price history, returning a DataFrame identical in
    shape to the old ``yf.Ticker.history()`` output.

    Parameters
    ----------
    client:
        Authenticated ``RESTClient``.
    ticker:
        US equity symbol (e.g. ``"AAPL"``).
    period:
        Lookback window expressed as a portfolio-builder-style string
        (``"1mo"``, ``"3mo"``, ``"6mo"``, ``"1y"``, ``"2y"``, ``"5y"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``Open, High, Low, Close, Volume``.
        Index: ``DatetimeIndex`` named ``"Date"``.
    """
    df, _status = fetch_price_history_with_status(
        client=client,
        ticker=ticker,
        period=period,
    )
    return df


# ---------------------------------------------------------------------------
# High-level convenience wrapper (replaces the old ``fetch_stock_data``)
# ---------------------------------------------------------------------------


def fetch_stock_data(
    client: RESTClient,
    ticker: str,
    history_period: str = "1y",
) -> dict[str, Any]:
    """Fetch all stock data for a single ticker.

    Returns
    -------
    dict
        ``{"history": pd.DataFrame, "history_status": str}``
    """
    history, history_status = fetch_price_history_with_status(
        client=client,
        ticker=ticker,
        period=history_period,
    )
    return {
        "history": history,
        "history_status": history_status,
    }
