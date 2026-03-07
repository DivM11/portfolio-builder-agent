"""Data client for fetching stock data from Massive.com (formerly Polygon.io)."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from massive import RESTClient

logger = logging.getLogger(__name__)

HISTORY_STATUS_OK = "ok"
HISTORY_STATUS_RATE_LIMITED = "rate_limited"
HISTORY_STATUS_NOT_FOUND = "not_found"
HISTORY_STATUS_EMPTY_DATA = "empty_data"
HISTORY_STATUS_UNEXPECTED_ERROR = "unexpected_error"

# yfinance-style period strings → number of calendar days
_PERIOD_DAYS: Dict[str, int] = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}

# Mapping: yfinance metric name → Massive.com income_statement attribute
_INCOME_METRIC_MAP: Dict[str, str] = {
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
) -> Tuple[pd.DataFrame, str]:
    """Fetch daily OHLCV price history plus a normalized fetch status."""
    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    days = _PERIOD_DAYS.get(period, 365)
    to_date = date.today()
    from_date = to_date - timedelta(days=days)

    try:
        aggs: List[Any] = list(client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_date.isoformat(),
            to=to_date.isoformat(),
            adjusted=True,
            sort="asc",
            limit=50000,
        ))
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
        Lookback window expressed as a yfinance-style string
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
# Financials (income statement sourced from SEC filings)
# ---------------------------------------------------------------------------

def _safe_metric_value(obj: Any, attr: str) -> Optional[float]:
    """Extract ``value`` from a Massive.com financials field safely."""
    field = getattr(obj, attr, None)
    if field is None:
        return None
    if isinstance(field, dict):
        return field.get("value")
    return getattr(field, "value", None)


def _compute_ebitda(
    income_statement: Any,
    cash_flow_statement: Any,
) -> Optional[float]:
    """Compute EBITDA = Operating Income + Depreciation & Amortization.

    EBITDA is not always a direct line item in SEC filings, so we compute
    it from available data when possible.
    """
    operating_income = _safe_metric_value(income_statement, "operating_income_loss")
    depreciation = _safe_metric_value(cash_flow_statement, "depreciation_and_amortization")
    if operating_income is not None and depreciation is not None:
        return operating_income + abs(depreciation)
    return operating_income  # fallback: just operating income if D&A unavailable


def fetch_financials(
    client: RESTClient,
    ticker: str,
    period: str = "annual",
) -> pd.DataFrame:
    """Fetch income-statement financials from Massive.com.

    Returns a DataFrame shaped like the old yfinance ``stock.financials``
    output: **rows = metric names**, **columns = period dates** (as
    ``Timestamp``).
    """
    timeframe = "quarterly" if period == "quarterly" else "annual"

    try:
        results: List[Any] = list(client.vx.list_stock_financials(
            ticker=ticker,
            timeframe=timeframe,
            limit=10,
            order="desc",
            sort="period_of_report_date",
        ))
    except Exception:
        logger.exception("Massive.com list_stock_financials failed for %s", ticker)
        return pd.DataFrame()

    if not results:
        return pd.DataFrame()

    records: Dict[str, Dict[str, Optional[float]]] = {}
    for filing in results:
        col_date = str(filing.period_of_report_date)
        inc = getattr(filing.financials, "income_statement", None)
        cf = getattr(filing.financials, "cash_flow_statement", None)

        row: Dict[str, Optional[float]] = {}
        if inc is not None:
            for yf_name, massive_attr in _INCOME_METRIC_MAP.items():
                row[yf_name] = _safe_metric_value(inc, massive_attr)
        row["EBITDA"] = _compute_ebitda(inc, cf) if inc is not None else None
        records[col_date] = row

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)  # columns = date strings, rows = metrics
    df.columns = pd.to_datetime(df.columns)
    df = df.sort_index(axis=1)
    return df


# ---------------------------------------------------------------------------
# High-level convenience wrapper (replaces the old ``fetch_stock_data``)
# ---------------------------------------------------------------------------

def fetch_stock_data(
    client: RESTClient,
    ticker: str,
    history_period: str = "1y",
    financials_period: str = "annual",
) -> Dict[str, Any]:
    """Fetch all stock data for a single ticker.

    Returns
    -------
    dict
        ``{"history": pd.DataFrame, "financials": pd.DataFrame, "history_status": str}``
    """
    history, history_status = fetch_price_history_with_status(
        client=client,
        ticker=ticker,
        period=history_period,
    )
    return {
        "history": history,
        "financials": pd.DataFrame(),
        "history_status": history_status,
    }
