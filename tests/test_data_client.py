"""Unit tests for the Massive.com data client module."""

from src.data_client import (
    HISTORY_STATUS_NOT_FOUND,
    HISTORY_STATUS_RATE_LIMITED,
    HISTORY_STATUS_UNEXPECTED_ERROR,
    create_massive_client,
    fetch_price_history,
    fetch_price_history_with_status,
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


class DummyVx:
    pass


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
    pass


class ErrorClient:
    def __init__(self, message: str):
        self.vx = DummyVx()
        self._message = message

    def list_aggs(self, **_kwargs):
        raise RuntimeError(self._message)


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


def test_fetch_price_history_with_status_rate_limited() -> None:
    client = ErrorClient("429 too many requests")

    df, status = fetch_price_history_with_status(client, "AAPL", period="1y")

    assert df.empty
    assert status == HISTORY_STATUS_RATE_LIMITED


def test_fetch_price_history_with_status_not_found() -> None:
    client = ErrorClient("404 not found")

    df, status = fetch_price_history_with_status(client, "MISSING", period="1y")

    assert df.empty
    assert status == HISTORY_STATUS_NOT_FOUND


def test_fetch_price_history_with_status_unexpected_error() -> None:
    client = ErrorClient("connection reset")

    df, status = fetch_price_history_with_status(client, "AAPL", period="1y")

    assert df.empty
    assert status == HISTORY_STATUS_UNEXPECTED_ERROR


# ---------------------------------------------------------------------------
# Tests — high-level fetch_stock_data
# ---------------------------------------------------------------------------


def test_fetch_stock_data_returns_all_keys():
    client = DummyMassiveClient()
    data = fetch_stock_data(client, "AAPL", history_period="1y")

    assert set(data.keys()) == {"history", "history_status"}
    assert not data["history"].empty
    assert data["history_status"] == "ok"


def test_create_massive_client_calls_rest_client(monkeypatch):
    class FakeRESTClient:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr("src.data_client.RESTClient", FakeRESTClient)

    client = create_massive_client("test-key")
    assert client.api_key == "test-key"
