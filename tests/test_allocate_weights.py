"""Unit tests for the allocate_weights tool."""

from src.tools.allocate_weights import allocate_weights_tool, tool_definition

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


def test_tool_definition_has_expected_name() -> None:
    definition = tool_definition()

    assert definition["type"] == "function"
    assert definition["function"]["name"] == "allocate_weights"


# ---------------------------------------------------------------------------
# Tool behavior
# ---------------------------------------------------------------------------


def test_allocate_weights_tool_normalizes_and_allocates() -> None:
    payload = allocate_weights_tool(
        {
            "tickers": ["aapl", "msft"],
            "weights": {"aapl": 0.6, "msft": 0.4},
            "portfolio_size": 1000,
        }
    )

    assert payload["normalized_weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    assert abs(payload["allocation"]["AAPL"] - 600.0) < 0.01
    assert abs(payload["allocation"]["MSFT"] - 400.0) < 0.01
    assert payload["portfolio_size"] == 1000.0


def test_allocate_weights_tool_handles_non_dict_weights() -> None:
    payload = allocate_weights_tool(
        {
            "tickers": ["AAPL", "MSFT"],
            "weights": "invalid",
            "portfolio_size": 100,
        }
    )

    assert payload["normalized_weights"] == {"AAPL": 0.5, "MSFT": 0.5}
    assert abs(payload["allocation"]["AAPL"] - 50.0) < 0.01
    assert abs(payload["allocation"]["MSFT"] - 50.0) < 0.01
