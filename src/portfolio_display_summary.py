"""Display formatting for portfolio and suggested changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PortfolioDisplaySummary:
    def format_suggestions(self, suggestions: Dict[str, object]) -> str:
        add = suggestions.get("add", []) if isinstance(suggestions, dict) else []
        remove = suggestions.get("remove", []) if isinstance(suggestions, dict) else []
        reweight = suggestions.get("reweight", {}) if isinstance(suggestions, dict) else {}

        if not add and not remove and not reweight:
            return "No suggested changes."

        lines: List[str] = ["Suggested portfolio changes:"]
        lines.append(f"- Add: {', '.join(add) if add else 'None'}")
        lines.append(f"- Remove: {', '.join(remove) if remove else 'None'}")

        if reweight:
            ordered = ", ".join(f"{ticker}: {weight:.2%}" for ticker, weight in reweight.items())
            lines.append(f"- Reweight: {ordered}")
        else:
            lines.append("- Reweight: None")

        return "\n".join(lines)

    def format_portfolio_header(self, tickers: List[str]) -> str:
        return "Recommended Portfolio Tickers: " + ", ".join(tickers) if tickers else "Recommended Portfolio Tickers: (none)"
