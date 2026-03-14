"""Single reasoning portfolio agent with explicit tool calling."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from src.agent_models import AgentContext, AgentResult
from src.data_client import create_massive_client, fetch_stock_data
from src.event_store.base import EventStore, NullEventStore
from src.event_store.models import EventRecord
from src.llm_service import LLMService
from src.tickr_data_manager import ProgressCallback, TickrDataManager
from src.tickr_summary_manager import TickrSummaryManager
from src.tools.allocate_weights import allocate_weights_tool
from src.tools.analyze_portfolio import analyze_portfolio_tool
from src.tools.build_summary import build_summary_tool
from src.tools.fetch_ticker_data import fetch_ticker_data_tool
from src.tools.generate_tickers import generate_tickers_tool
from src.tools.allocate_weights import tool_definition as allocate_weights_tool_definition
from src.tools.analyze_portfolio import tool_definition as analyze_portfolio_tool_definition
from src.tools.build_summary import tool_definition as build_summary_tool_definition
from src.tools.fetch_ticker_data import tool_definition as fetch_ticker_data_tool_definition
from src.tools.generate_tickers import tool_definition as generate_tickers_tool_definition

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """Single portfolio agent that iterates through tool calls."""

    def __init__(
        self,
        llm_service: LLMService,
        config: dict[str, Any],
        *,
        event_store: EventStore | None = None,
        tickr_data_manager: TickrDataManager | None = None,
        tickr_summary_manager: TickrSummaryManager | None = None,
        massive_client_factory: Callable[[str], Any] = create_massive_client,
        stock_data_fetcher: Callable[..., dict[str, Any]] = fetch_stock_data,
    ) -> None:
        self.llm_service = llm_service
        self.config = config
        self.event_store = event_store or NullEventStore()
        self.tickr_data_manager = tickr_data_manager or TickrDataManager()
        self.tickr_summary_manager = tickr_summary_manager or TickrSummaryManager()
        self._massive_client_factory = massive_client_factory
        self._stock_data_fetcher = stock_data_fetcher
        self._messages: list[dict[str, Any]] = []
        self._last_result = AgentResult()

    def run(
        self,
        *,
        user_input: str,
        portfolio_size: float,
        excluded_tickers: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        context = AgentContext(
            user_input=user_input,
            portfolio_size=portfolio_size,
            excluded_tickers=tuple(excluded_tickers or []),
            session_id=session_id,
            run_id=run_id,
        )
        self._messages = [
            {"role": "system", "content": self._system_prompt(context)},
            {"role": "user", "content": user_input},
        ]
        self._record_event(
            "agent_start",
            context,
            action_payload={"user_input": user_input, "portfolio_size": portfolio_size},
        )
        self._last_result = self._run_loop(context, progress_callback=progress_callback)
        return self._last_result

    def refine(
        self,
        *,
        feedback: str,
        session_id: str | None = None,
        run_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        if not self._messages:
            raise ValueError("Cannot refine before running the agent at least once")

        previous = self._last_result
        context = AgentContext(
            user_input=feedback,
            portfolio_size=float(sum(previous.allocation.values())) if previous.allocation else 0.0,
            excluded_tickers=tuple(previous.metadata.get("excluded_tickers", [])),
            session_id=session_id,
            run_id=run_id,
        )
        self._messages.append({"role": "user", "content": feedback})
        self._record_event(
            "user_action",
            context,
            action="refine",
            action_payload={"feedback": feedback},
        )
        self._last_result = self._run_loop(context, progress_callback=progress_callback)
        return self._last_result

    def _system_prompt(self, context: AgentContext) -> str:
        agent_cfg = self.config.get("agent", {})
        prompt_template = agent_cfg.get("system_prompt")
        if prompt_template:
            return prompt_template.format(
                max_tickers=self.config.get("stocks", {}).get("max_tickers", 10),
                excluded_tickers=", ".join(context.excluded_tickers),
            )
        return (
            "You are a portfolio building agent. Build a US equities portfolio by calling tools. "
            "Call tools in sequence: generate_tickers, fetch_ticker_data, build_summary, "
            "allocate_weights, analyze_portfolio. When done, return strict JSON with keys: "
            "tickers, weights, allocation, analysis_text, suggestions."
        )

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            generate_tickers_tool_definition(),
            fetch_ticker_data_tool_definition(),
            build_summary_tool_definition(),
            allocate_weights_tool_definition(),
            analyze_portfolio_tool_definition(),
        ]

    def _run_loop(
        self,
        context: AgentContext,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        openrouter_cfg = self.config.get("openrouter", {})
        agent_cfg = self.config.get("agent", {})
        model = agent_cfg.get(
            "model",
            openrouter_cfg.get("default_models", {}).get("ticker", "anthropic/claude-sonnet-4.5"),
        )
        max_tokens = int(agent_cfg.get("max_tokens", 4096))
        temperature = float(agent_cfg.get("temperature", 0.2))
        max_tool_rounds = int(agent_cfg.get("max_tool_rounds", 10))
        schema_version = int(self.config.get("event_store", {}).get("schema_version", 1))

        massive_cfg = self.config.get("massive", {}).get("api", {})
        massive_api_key = massive_cfg.get("api_key")
        if not massive_api_key:
            raise ValueError("Missing Massive.com API key")
        massive_client = self._massive_client_factory(massive_api_key)

        for round_index in range(max_tool_rounds):
            response = self.llm_service.complete_with_tools(
                request_name="portfolio_agent",
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=self._messages,
                tools=self._tool_definitions(),
                session_id=context.session_id,
                run_id=context.run_id,
            )

            if response.has_tool_calls:
                assistant_tool_message = {
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
                self._messages.append(assistant_tool_message)
                for call in response.tool_calls:
                    result_payload = self._execute_tool(
                        call.name,
                        call.arguments,
                        context=context,
                        progress_callback=progress_callback,
                        massive_client=massive_client,
                    )
                    self.event_store.record(
                        EventRecord(
                            event_type="tool_result",
                            schema_version=schema_version,
                            session_id=context.session_id or "n/a",
                            run_id=context.run_id or "n/a",
                            tool_name=call.name,
                            tool_arguments=call.arguments,
                            tool_result=result_payload,
                            tool_call_id=call.id,
                            agent_round=round_index + 1,
                        )
                    )
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": json.dumps(result_payload),
                        }
                    )
                continue

            self._messages.append({"role": "assistant", "content": response.text or ""})
            result = self._parse_final_result(response.text or "")
            result.messages = list(self._messages)
            if "excluded_tickers" not in result.metadata:
                result.metadata["excluded_tickers"] = list(context.excluded_tickers)
            self._record_event(
                "agent_complete",
                context,
                action_payload={"tickers": result.tickers, "weights": result.weights},
            )
            return result

        raise ValueError("Agent exceeded max tool rounds without producing a final response")

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: AgentContext,
        progress_callback: ProgressCallback | None,
        massive_client: Any,
    ) -> dict[str, Any]:
        self._record_event(
            "tool_call",
            context,
            action_payload={"tool_name": name, "arguments": arguments},
        )
        stocks_cfg = self.config.get("stocks", {})
        if name == "generate_tickers":
            payload = generate_tickers_tool(
                arguments,
                max_tickers=int(stocks_cfg.get("max_tickers", 10)),
            )
            excluded = set(context.excluded_tickers)
            filtered = [ticker for ticker in payload["valid_tickers"] if ticker not in excluded]
            payload["valid_tickers"] = filtered
            payload["count"] = len(filtered)
            return payload

        if name == "fetch_ticker_data":
            return fetch_ticker_data_tool(
                arguments,
                tickr_data_manager=self.tickr_data_manager,
                stock_data_fetcher=self._stock_data_fetcher,
                history_period=stocks_cfg.get("history_period", "1y"),
                financials_period=stocks_cfg.get("financials_period", "quarterly"),
                massive_client=massive_client,
                progress_callback=progress_callback,
            )

        if name == "build_summary":
            return build_summary_tool(
                arguments,
                tickr_data_manager=self.tickr_data_manager,
                tickr_summary_manager=self.tickr_summary_manager,
                financial_metrics=stocks_cfg.get("financials_metrics", []),
            )

        if name == "allocate_weights":
            return allocate_weights_tool(arguments)

        if name == "analyze_portfolio":
            return analyze_portfolio_tool(
                arguments,
                tickr_data_manager=self.tickr_data_manager,
                financial_metrics=stocks_cfg.get("financials_metrics", []),
            )

        return {"error": f"Unknown tool: {name}"}

    def _parse_final_result(self, text: str) -> AgentResult:
        payload = _extract_json_payload(text)
        tickers = [str(item).upper() for item in payload.get("tickers", []) if str(item).strip()]
        weights_raw = payload.get("weights", {})
        if not isinstance(weights_raw, dict):
            weights_raw = {}
        weights = {str(k).upper(): float(v) for k, v in weights_raw.items()}

        allocation_raw = payload.get("allocation", {})
        if not isinstance(allocation_raw, dict):
            allocation_raw = {}
        allocation = {str(k).upper(): float(v) for k, v in allocation_raw.items()}

        suggestions = payload.get("suggestions", {})
        if not isinstance(suggestions, dict):
            suggestions = {}

        analysis_text = str(payload.get("analysis_text", text)).strip()
        data_by_ticker = self.tickr_data_manager.get_data_by_ticker(tickers)
        summary_text = self.tickr_summary_manager.build_or_get_summary(
            tickers=tickers,
            data_by_ticker=data_by_ticker,
            financial_metrics=self.config.get("stocks", {}).get("financials_metrics", []),
            data_version=self.tickr_data_manager.cache_version,
        )

        return AgentResult(
            tickers=tickers,
            data_by_ticker=data_by_ticker,
            summary_text=summary_text,
            weights=weights,
            allocation=allocation,
            analysis_text=analysis_text,
            suggestions=suggestions,
            metadata={
                "raw_final_output": text,
                "parsed_json": payload,
                "excluded_tickers": payload.get("excluded_tickers", []),
            },
        )

    def _record_event(
        self,
        event_type: str,
        context: AgentContext,
        *,
        action: str | None = None,
        action_payload: dict[str, Any] | None = None,
    ) -> None:
        schema_version = int(self.config.get("event_store", {}).get("schema_version", 1))
        self.event_store.record(
            EventRecord(
                event_type=event_type,
                schema_version=schema_version,
                session_id=context.session_id or "n/a",
                run_id=context.run_id or "n/a",
                action=action,
                action_payload=action_payload,
                agent="portfolio_agent",
            )
        )


def _extract_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[1]

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}
