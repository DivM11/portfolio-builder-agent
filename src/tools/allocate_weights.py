"""Weight allocation tool used by the portfolio agent."""

from __future__ import annotations

from typing import Any

from src.portfolio import allocate_portfolio_by_weights, normalize_weights


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "allocate_weights",
            "description": "Normalize weights and convert them into a dollar allocation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weights": {
                        "type": "object",
                        "description": "Ticker-to-weight map.",
                    },
                    "portfolio_size": {
                        "type": "number",
                        "description": "Portfolio size in USD.",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers to normalize against.",
                    },
                },
                "required": ["weights", "portfolio_size", "tickers"],
            },
        },
    }


def allocate_weights_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    tickers = [str(t).upper() for t in arguments.get("tickers", []) if str(t).strip()]
    weights_raw = arguments.get("weights", {})
    if not isinstance(weights_raw, dict):
        weights_raw = {}
    weights = {str(ticker).upper(): float(value) for ticker, value in weights_raw.items() if str(ticker).strip()}
    portfolio_size = float(arguments.get("portfolio_size", 0.0))
    normalized = normalize_weights(weights, tickers)
    allocation = allocate_portfolio_by_weights(tickers, portfolio_size, normalized)
    return {
        "normalized_weights": normalized,
        "allocation": allocation,
        "portfolio_size": portfolio_size,
    }
