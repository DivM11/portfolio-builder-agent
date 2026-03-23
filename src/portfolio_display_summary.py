"""Display formatting for portfolio and suggested changes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioDisplaySummary:
    def format_suggestions(self, suggestions: dict[str, object]) -> str:
        add_raw = suggestions.get("add", []) if isinstance(suggestions, dict) else []
        remove_raw = suggestions.get("remove", []) if isinstance(suggestions, dict) else []
        reweight_raw = suggestions.get("reweight", {}) if isinstance(suggestions, dict) else {}

        add = [str(item).upper() for item in add_raw] if isinstance(add_raw, list) else []
        remove = [str(item).upper() for item in remove_raw] if isinstance(remove_raw, list) else []
        reweight: dict[str, float] = {}
        if isinstance(reweight_raw, dict):
            for ticker, weight in reweight_raw.items():
                try:
                    reweight[str(ticker).upper()] = float(weight)
                except (TypeError, ValueError):
                    continue

        if not add and not remove and not reweight:
            return "No suggested changes."

        lines: list[str] = ["Suggested portfolio changes:"]
        lines.append(f"- Add: {', '.join(add) if add else 'None'}")
        lines.append(f"- Remove: {', '.join(remove) if remove else 'None'}")

        if reweight:
            ordered = ", ".join(f"{ticker}: {weight:.2%}" for ticker, weight in reweight.items())
            lines.append(f"- Reweight: {ordered}")
        else:
            lines.append("- Reweight: None")

        return "\n".join(lines)

    def format_portfolio_header(self, tickers: list[str]) -> str:
        return (
            "Recommended Portfolio Tickers: " + ", ".join(tickers)
            if tickers
            else "Recommended Portfolio Tickers: (none)"
        )
