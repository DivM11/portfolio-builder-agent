"""Pydantic schemas for API-boundary validation of data frames and dicts.

Validation is intentionally lightweight — we check column presence and
numeric types at the two external boundaries:
  1. ``fetch_price_history_with_status`` (data arriving from Massive.com)
  2. ``summarize_portfolio_stats`` (computed stats leaving the analytics layer)

Internal transformation steps are *not* validated here to avoid overhead.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name constants (single source of truth, replaces scattered strings)
# ---------------------------------------------------------------------------

OHLCV_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]
OHLCV_INDEX_NAME: str = "Date"

ALLOCATION_COLUMNS: list[str] = ["Ticker", "Weight", "Allocation"]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class OHLCVRowSchema(BaseModel):
    """Validates a single row of OHLCV price history."""

    Open: float
    High: float
    Low: float
    Close: float
    Volume: float

    @field_validator("Open", "High", "Low", "Close", "Volume", mode="before")
    @classmethod
    def must_be_finite(cls, v: Any) -> float:
        import math

        value = float(v)
        if not math.isfinite(value):
            raise ValueError(f"Expected finite number, got {v!r}")
        return value


class PortfolioStatsSchema(BaseModel):
    """Validates the dict returned by ``summarize_portfolio_stats``."""

    min: float
    max: float
    median: float
    current: float
    return_1y: float

    @field_validator("min", "max", "median", "current", "return_1y", mode="before")
    @classmethod
    def must_be_finite(cls, v: Any) -> float:
        import math

        value = float(v)
        if not math.isfinite(value):
            raise ValueError(f"Expected finite number, got {v!r}")
        return value


class AllocationRowSchema(BaseModel):
    """Validates a single row of the portfolio allocation display table."""

    Ticker: str
    Weight: float
    Allocation: float


# ---------------------------------------------------------------------------
# DataFrame validator
# ---------------------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame, schema: type[BaseModel]) -> list[str]:
    """Validate a DataFrame against a Pydantic schema row-by-row.

    Returns a list of human-readable error strings (empty == valid).
    Validation is non-destructive — the DataFrame is never mutated.
    """
    errors: list[str] = []

    expected_columns = set(schema.model_fields.keys())
    missing = expected_columns - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        return errors  # can't validate rows without required columns

    for idx, row in df.iterrows():
        try:
            schema(**{col: row[col] for col in expected_columns})
        except Exception as exc:
            errors.append(f"Row {idx}: {exc}")

    return errors


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Validate that a DataFrame conforms to the OHLCV schema.

    Checks column presence, numeric types, and finiteness.
    Returns validation error strings (empty list == valid).
    """
    return validate_dataframe(df, OHLCVRowSchema)


def validate_portfolio_stats(stats: dict[str, Any]) -> list[str]:
    """Validate a portfolio stats dict against ``PortfolioStatsSchema``.

    Returns validation error strings (empty list == valid).
    """
    errors: list[str] = []
    try:
        PortfolioStatsSchema(**stats)
    except Exception as exc:
        errors.append(str(exc))
    return errors
