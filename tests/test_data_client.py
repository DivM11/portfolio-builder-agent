"""Unit tests for the Massive.com data client module."""

import pandas as pd

from src.data_client import (
    create_massive_client,
    fetch_financials,
    fetch_price_history,
    fetch_stock_data,
)


# ---------------------------------------------------------------------------
# Stubs for Massive.com SDK objects
# ---------------------------------------------------------------------------

class DummyAgg:
    def __init__(self, o, h, l, c, v, ts):  # noqa: E741
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.timestamp = ts


class DummyMetricField:
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
    def list_stock_financials(self, **_kwargs):
        return [DummyStockFinancial("2025-12-31"), DummyStockFinancial("2025-09-30")]


class DummyMassiveClient:
    def __init__(self):
        self.vx = DummyVx()

    def list_aggs(self, **_kwargs):
        return [
            DummyAgg(100, 105, 99, 102, 1000, 1700000000000),
            DummyAgg(102, 106, 101, 104, 1200, 1700086400000),
            DummyAgg(104, 108, 103, 107, 1100, 1700172800000),
        ]


class DummyEmptyClient:
    """Client that returns empty results."""

    def __init__(self):
        self.vx = DummyEmptyVx()

    def list_aggs(self, **_kwargs):
        return []


class DummyEmptyVx:
    def list_stock_financials(self, **_kwargs):
        return []


class DummyNoFinancialsVx:
    def list_stock_financials(self, **_kwargs):
        raise AssertionError("Financials endpoint should not be called")


class DummyNoFinancialsClient:
    def __init__(self):
        self.vx = DummyNoFinancialsVx()

    def list_aggs(self, **_kwargs):
        return [
            DummyAgg(100, 105, 99, 102, 1000, 1700000000000),
            DummyAgg(102, 106, 101, 104, 1200, 1700086400000),
        ]


# ---------------------------------------------------------------------------
# Tests — price history
# ---------------------------------------------------------------------------

def test_fetch_price_history_returns_ohlcv():
    client = DummyMassiveClient()
    df = fetch_price_history(client, "AAPL", period="1y")

    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert len(df) == 3
    assert df["Close"].iloc[0] == 102


def test_fetch_price_history_empty_aggs():
    client = DummyEmptyClient()
    df = fetch_price_history(client, "AAPL", period="1y")

    assert df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetch_price_history_unknown_period():
    """Unknown period string should default to 365 days."""
    client = DummyMassiveClient()
    df = fetch_price_history(client, "AAPL", period="10y")

    assert not df.empty


# ---------------------------------------------------------------------------
# Tests — financials
# ---------------------------------------------------------------------------

def test_fetch_financials_returns_metric_rows():
    client = DummyMassiveClient()
    df = fetch_financials(client, "AAPL", period="annual")

    assert not df.empty
    assert "Total Revenue" in df.index
    assert "Net Income" in df.index
    assert "EBITDA" in df.index
    assert len(df.columns) == 2  # two filing periods


def test_fetch_financials_ebitda_computation():
    client = DummyMassiveClient()
    df = fetch_financials(client, "AAPL", period="annual")

    # EBITDA = operating income (300_000) + abs(depreciation (50_000)) = 350_000
    ebitda = df.loc["EBITDA"].iloc[0]
    assert ebitda == 350_000


def test_fetch_financials_empty():
    client = DummyEmptyClient()
    df = fetch_financials(client, "AAPL", period="quarterly")

    assert df.empty


# ---------------------------------------------------------------------------
# Tests — high-level fetch_stock_data
# ---------------------------------------------------------------------------

def test_fetch_stock_data_returns_all_keys():
    client = DummyMassiveClient()
    data = fetch_stock_data(client, "AAPL", history_period="1y")

    assert set(data.keys()) == {"history", "financials", "history_status"}
    assert not data["history"].empty
    assert data["financials"].empty
    assert data["history_status"] == "ok"


def test_fetch_stock_data_skips_financials_endpoint():
    client = DummyNoFinancialsClient()

    data = fetch_stock_data(client, "AAPL", history_period="1y")

    assert not data["history"].empty
    assert data["financials"].empty
    assert data["history_status"] == "ok"


def test_create_massive_client_calls_rest_client(monkeypatch):
    class FakeRESTClient:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr("src.data_client.RESTClient", FakeRESTClient)

    client = create_massive_client("test-key")
    assert client.api_key == "test-key"
