"""Ticker validation tool used by the portfolio agent."""

from __future__ import annotations

from typing import Any

from src.llm_validation import extract_valid_tickers


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "generate_tickers",
            "description": "Validate and normalize candidate US stock tickers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate ticker symbols.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why these tickers were chosen.",
                    },
                },
                "required": ["tickers"],
            },
        },
    }


def generate_tickers_tool(arguments: dict[str, Any], *, max_tickers: int) -> dict[str, Any]:
    raw = arguments.get("tickers", [])
    if not isinstance(raw, list):
        raw = []
    joined = ",".join(str(ticker) for ticker in raw)
    valid = extract_valid_tickers(joined, delimiter=",")
    valid = valid[: max(0, int(max_tickers))]
    rejected = [str(t).upper() for t in raw if str(t).upper() not in valid]
    raw_reasoning = str(arguments.get("reasoning", ""))
    # Sanitize: truncate and strip control characters to prevent reflected injection
    sanitized_reasoning = "".join(ch for ch in raw_reasoning if ch >= " " or ch in "\n\t")[:500]
    return {
        "valid_tickers": valid,
        "rejected_tickers": rejected,
        "count": len(valid),
        "reasoning": sanitized_reasoning,
    }
