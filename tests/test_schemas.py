"""Tests for src/schemas.py — Pydantic boundary validation."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from src.schemas import (
    ALLOCATION_COLUMNS,
    OHLCV_COLUMNS,
    AllocationRowSchema,
    OHLCVRowSchema,
    PortfolioStatsSchema,
    validate_dataframe,
    validate_ohlcv,
    validate_portfolio_stats,
)

# ---------------------------------------------------------------------------
# Column-name constants
# ---------------------------------------------------------------------------


def test_ohlcv_columns_constant() -> None:
    assert OHLCV_COLUMNS == ["Open", "High", "Low", "Close", "Volume"]


def test_allocation_columns_constant() -> None:
    assert ALLOCATION_COLUMNS == ["Ticker", "Weight", "Allocation"]


# ---------------------------------------------------------------------------
# OHLCVRowSchema
# ---------------------------------------------------------------------------


def test_ohlcv_row_schema_valid() -> None:
    row = OHLCVRowSchema(Open=100.0, High=110.0, Low=95.0, Close=105.0, Volume=1_000_000.0)
    assert row.Close == 105.0


def test_ohlcv_row_schema_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        OHLCVRowSchema(Open=float("nan"), High=110.0, Low=95.0, Close=105.0, Volume=1e6)


def test_ohlcv_row_schema_rejects_inf() -> None:
    with pytest.raises(ValidationError):
        OHLCVRowSchema(Open=100.0, High=float("inf"), Low=95.0, Close=105.0, Volume=1e6)


# ---------------------------------------------------------------------------
# PortfolioStatsSchema
# ---------------------------------------------------------------------------


def test_portfolio_stats_schema_valid() -> None:
    stats = PortfolioStatsSchema(min=0.9, max=1.2, median=1.05, current=1.1, return_1y=0.1)
    assert stats.return_1y == 0.1


def test_portfolio_stats_schema_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        PortfolioStatsSchema(min=float("nan"), max=1.2, median=1.05, current=1.1, return_1y=0.1)


def test_portfolio_stats_schema_rejects_missing_field() -> None:
    with pytest.raises(ValidationError):
        PortfolioStatsSchema(min=0.9, max=1.2, median=1.05, current=1.1)  # return_1y missing


# ---------------------------------------------------------------------------
# AllocationRowSchema
# ---------------------------------------------------------------------------


def test_allocation_row_schema_valid() -> None:
    row = AllocationRowSchema(Ticker="AAPL", Weight=0.6, Allocation=600.0)
    assert row.Ticker == "AAPL"


# ---------------------------------------------------------------------------
# validate_ohlcv
# ---------------------------------------------------------------------------


def _make_ohlcv(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0] * rows,
            "High": [110.0] * rows,
            "Low": [95.0] * rows,
            "Close": [105.0] * rows,
            "Volume": [1_000_000.0] * rows,
        }
    )


def test_validate_ohlcv_valid_returns_no_errors() -> None:
    errors = validate_ohlcv(_make_ohlcv())
    assert errors == []


def test_validate_ohlcv_missing_column_returns_error() -> None:
    df = _make_ohlcv().drop(columns=["Close"])
    errors = validate_ohlcv(df)
    assert any("Close" in e for e in errors)


def test_validate_ohlcv_multiple_missing_columns() -> None:
    df = _make_ohlcv().drop(columns=["Close", "Volume"])
    errors = validate_ohlcv(df)
    assert errors  # at least one error
    missing_report = errors[0]
    assert "Close" in missing_report or "Volume" in missing_report


def test_validate_ohlcv_nan_value_returns_error() -> None:
    df = _make_ohlcv()
    df.loc[1, "Close"] = float("nan")
    errors = validate_ohlcv(df)
    assert errors  # row 1 should fail


def test_validate_ohlcv_inf_value_returns_error() -> None:
    df = _make_ohlcv()
    df.loc[0, "High"] = float("inf")
    errors = validate_ohlcv(df)
    assert errors


def test_validate_ohlcv_empty_dataframe_returns_no_errors() -> None:
    df = pd.DataFrame(columns=OHLCV_COLUMNS)
    errors = validate_ohlcv(df)
    assert errors == []


# ---------------------------------------------------------------------------
# validate_portfolio_stats
# ---------------------------------------------------------------------------


def _valid_stats() -> dict:
    return {"min": 0.9, "max": 1.2, "median": 1.05, "current": 1.1, "return_1y": 0.1}


def test_validate_portfolio_stats_valid() -> None:
    assert validate_portfolio_stats(_valid_stats()) == []


def test_validate_portfolio_stats_missing_key() -> None:
    stats = _valid_stats()
    del stats["return_1y"]
    errors = validate_portfolio_stats(stats)
    assert errors


def test_validate_portfolio_stats_nan_value() -> None:
    stats = _valid_stats()
    stats["current"] = float("nan")
    errors = validate_portfolio_stats(stats)
    assert errors


def test_validate_portfolio_stats_empty_dict() -> None:
    errors = validate_portfolio_stats({})
    assert errors


# ---------------------------------------------------------------------------
# validate_dataframe generic
# ---------------------------------------------------------------------------


def test_validate_dataframe_valid() -> None:
    df = _make_ohlcv()
    errors = validate_dataframe(df, OHLCVRowSchema)
    assert errors == []


def test_validate_dataframe_wrong_type_coerced() -> None:
    """String numerics that are coercible should pass."""
    df = _make_ohlcv()
    df["Close"] = df["Close"].astype(str)  # "105.0" → Pydantic coerces to float
    errors = validate_dataframe(df, OHLCVRowSchema)
    assert errors == []


def test_validate_dataframe_non_numeric_fails() -> None:
    df = _make_ohlcv().astype(object)  # object dtype accepts mixed values without FutureWarning
    df.loc[0, "Close"] = "not-a-number"
    errors = validate_dataframe(df, OHLCVRowSchema)
    assert errors  # row 0 should fail


# ---------------------------------------------------------------------------
# Integration: validate_ohlcv called inside fetch_price_history_with_status
# ---------------------------------------------------------------------------


def test_fetch_price_history_with_status_validates_ohlcv(monkeypatch) -> None:
    """Verify that fetch_price_history_with_status logs but does NOT raise
    on NaN data — validation is advisory, not blocking."""
    import logging

    from src.data_client import fetch_price_history_with_status

    class _FakeAgg:
        def __init__(self, i):
            self.open = 100.0
            self.high = 110.0
            self.low = 95.0
            self.close = float("nan") if i == 0 else 105.0
            self.volume = 1e6
            self.timestamp = 1_700_000_000_000 + i * 86_400_000

    class _FakeClient:
        def list_aggs(self, **_kwargs):
            return [_FakeAgg(i) for i in range(3)]

    warning_messages: list[str] = []

    class _CapHandler(logging.Handler):
        def emit(self, record):
            warning_messages.append(record.getMessage())

    handler = _CapHandler()
    logging.getLogger("src.data_client").addHandler(handler)
    try:
        df, status = fetch_price_history_with_status(_FakeClient(), "FAKE")
    finally:
        logging.getLogger("src.data_client").removeHandler(handler)

    # Should still return data (non-blocking)
    assert status == "ok"
    assert not df.empty
    # Warning should have been logged
    assert any("OHLCV validation" in m or "validation" in m.lower() for m in warning_messages)
