"""Single reasoning portfolio agent with explicit tool calling."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.agent_models import AgentContext, AgentResult, Context
from src.data_client import create_massive_client, fetch_stock_data
from src.etl.agent_performance import materialise_agent_performance
from src.event_store.base import EventStore, MonitoringStore, NullEventStore
from src.event_store.models import EventRecord, ToolCallRecord
from src.input_guard import InputGuard
from src.llm_service import LLMService
from src.tickr_data_manager import ProgressCallback, TickrDataManager
from src.tickr_summary_manager import TickrSummaryManager
from src.tools.allocate_weights import allocate_weights_tool
from src.tools.allocate_weights import tool_definition as allocate_weights_tool_definition
from src.tools.analyze_portfolio import analyze_portfolio_tool
from src.tools.analyze_portfolio import tool_definition as analyze_portfolio_tool_definition
from src.tools.build_summary import build_summary_tool
from src.tools.build_summary import tool_definition as build_summary_tool_definition
from src.tools.fetch_ticker_data import fetch_ticker_data_tool
from src.tools.fetch_ticker_data import tool_definition as fetch_ticker_data_tool_definition
from src.tools.generate_tickers import generate_tickers_tool
from src.tools.generate_tickers import tool_definition as generate_tickers_tool_definition

logger = logging.getLogger(__name__)
StepStatusCallback = Callable[[str], None]


class PortfolioAgent:
    """Single portfolio agent that iterates through tool calls."""

    def __init__(
        self,
        llm_service: LLMService,
        config: dict[str, Any],
        *,
        event_store: EventStore | None = None,
        input_guard: InputGuard | None = None,
        tickr_data_manager: TickrDataManager | None = None,
        tickr_summary_manager: TickrSummaryManager | None = None,
        massive_client_factory: Callable[[str], Any] = create_massive_client,
        stock_data_fetcher: Callable[..., dict[str, Any]] = fetch_stock_data,
    ) -> None:
        self.llm_service = llm_service
        self.config = config
        self.event_store = event_store or NullEventStore()
        self.input_guard = input_guard
        self.tickr_data_manager = tickr_data_manager or TickrDataManager()
        self.tickr_summary_manager = tickr_summary_manager or TickrSummaryManager()
        self._massive_client_factory = massive_client_factory
        self._stock_data_fetcher = stock_data_fetcher
        self._context: Context | None = None
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
        status_callback: StepStatusCallback | None = None,
    ) -> AgentResult:
        ctx = Context(
            user_input=user_input,
            portfolio_size=portfolio_size,
            excluded_tickers=excluded_tickers,
            session_id=session_id,
            run_id=run_id,
        )
        self._context = ctx

        if self.input_guard is not None:
            guard_result = self.input_guard.check(user_input, session_id=session_id, run_id=run_id)
            if not guard_result.safe:
                return AgentResult(
                    analysis_text=(
                        "I can only help with US equity portfolio questions. Please rephrase your request."
                        if guard_result.category == "off_topic"
                        else "Your message could not be processed. Please rephrase your request."
                    ),
                    metadata={
                        "guard_blocked": True,
                        "guard_category": guard_result.category,
                        "guard_reason": guard_result.reason,
                    },
                )

        agent_context = ctx.to_agent_context()
        ctx.prepare_for_run(self._system_prompt(agent_context))
        self._record_event(
            "agent_start",
            agent_context,
            action_payload={"user_input": user_input, "portfolio_size": portfolio_size},
        )
        self._last_result = self._run_loop(
            ctx,
            progress_callback=progress_callback,
            status_callback=status_callback,
            seed_result=None,
        )
        ctx.last_result = self._last_result
        self._emit_agent_performance(ctx)
        return self._last_result

    def refine(
        self,
        *,
        feedback: str,
        session_id: str | None = None,
        run_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        status_callback: StepStatusCallback | None = None,
    ) -> AgentResult:
        if self._context is None or not self._context.messages:
            raise ValueError("Cannot refine before running the agent at least once")

        if self.input_guard is not None:
            guard_result = self.input_guard.check(feedback, session_id=session_id, run_id=run_id)
            if not guard_result.safe:
                return AgentResult(
                    analysis_text=(
                        "I can only help with US equity portfolio questions. Please rephrase your request."
                        if guard_result.category == "off_topic"
                        else "Your message could not be processed. Please rephrase your request."
                    ),
                    metadata={
                        "guard_blocked": True,
                        "guard_category": guard_result.category,
                        "guard_reason": guard_result.reason,
                    },
                )

        previous = self._last_result
        ctx = self._context
        ctx.session_id = session_id
        ctx.run_id = run_id
        ctx.prepare_for_refine(feedback, previous)
        agent_context = ctx.to_agent_context()
        self._record_event(
            "user_action",
            agent_context,
            action="refine",
            action_payload={"feedback": feedback},
        )
        self._last_result = self._run_loop(
            ctx,
            progress_callback=progress_callback,
            status_callback=status_callback,
            seed_result=previous,
        )
        ctx.last_result = self._last_result
        self._emit_agent_performance(ctx)
        return self._last_result

    def _emit_agent_performance(self, ctx: Context) -> None:
        if not isinstance(self.event_store, MonitoringStore):
            return
        if not ctx.session_id or not ctx.run_id:
            return
        agent_cfg = self.config.get("agent", {})
        schema_version = int(self.config.get("event_store", {}).get("schema_version", 1))
        portfolio_stats: dict[str, Any] = {}
        analysis = ctx.work_state.get("analysis")
        if isinstance(analysis, dict):
            portfolio_stats = {k: analysis[k] for k in ("return_1y", "current", "min", "max") if k in analysis}
        try:
            materialise_agent_performance(
                self.event_store,
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                result=self._last_result,
                portfolio_stats=portfolio_stats,
                model=agent_cfg.get("model", ""),
                schema_version=schema_version,
                status="completed",
            )
        except Exception:
            logger.exception(
                "[session=%s run=%s] Failed to materialise agent performance",
                ctx.session_id,
                ctx.run_id,
            )

    def _system_prompt(self, context: AgentContext) -> str:
        agent_cfg = self.config.get("agent", {})
        prompt_template = agent_cfg.get("system_prompt")
        if isinstance(prompt_template, str) and prompt_template:
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
        ctx: Context,
        *,
        progress_callback: ProgressCallback | None = None,
        status_callback: StepStatusCallback | None = None,
        seed_result: AgentResult | None = None,
    ) -> AgentResult:
        agent_cfg = self.config.get("agent", {})
        model = agent_cfg.get("model", "anthropic/claude-sonnet-4.5")
        max_tokens = int(agent_cfg.get("max_tokens", 4096))
        temperature = float(agent_cfg.get("temperature", 0.2))
        max_tool_rounds = int(agent_cfg.get("max_tool_rounds", 10))
        reasoning_cfg = agent_cfg.get("reasoning")
        if not isinstance(reasoning_cfg, dict):
            reasoning_cfg = None
        schema_version = int(self.config.get("event_store", {}).get("schema_version", 1))

        massive_cfg = self.config.get("massive", {}).get("api", {})
        massive_api_key = massive_cfg.get("api_key")
        if not massive_api_key:
            raise ValueError("Missing Massive.com API key")
        massive_client = self._massive_client_factory(massive_api_key)

        agent_context = ctx.to_agent_context()
        work_state = ctx.work_state

        for round_index in range(max_tool_rounds):
            response = self.llm_service.complete_with_tools(
                request_name="portfolio_agent",
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=ctx.messages,
                tools=self._tool_definitions(),
                reasoning=reasoning_cfg,
                response_format=self._response_format(),
                session_id=ctx.session_id,
                run_id=ctx.run_id,
            )

            if response.has_tool_calls:
                if response.text:
                    work_state["reasoning_text"] = str(response.text).strip()
                tool_call_dicts = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in response.tool_calls
                ]
                ctx.add_assistant_tool_calls_message(response.text or "", tool_call_dicts)
                for call in response.tool_calls:
                    ctx.record_tool_invocation(call.name, call.arguments)
                    work_state["tool_invocations"].append({"name": call.name, "arguments": call.arguments})
                    result_payload = self._execute_tool(
                        call.name,
                        call.arguments,
                        context=agent_context,
                        progress_callback=progress_callback,
                        status_callback=status_callback,
                        massive_client=massive_client,
                        work_state=work_state,
                    )
                    self.event_store.record(
                        EventRecord(
                            event_type="tool_result",
                            schema_version=schema_version,
                            session_id=agent_context.session_id or "n/a",
                            run_id=agent_context.run_id or "n/a",
                            tool_name=call.name,
                            tool_arguments=call.arguments,
                            tool_result=result_payload,
                            tool_call_id=call.id,
                            agent_round=round_index + 1,
                        )
                    )
                    if isinstance(self.event_store, MonitoringStore):
                        self.event_store.record_tool_call(
                            ToolCallRecord(
                                id=str(uuid4()),
                                session_id=agent_context.session_id or "n/a",
                                run_id=agent_context.run_id or "n/a",
                                timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
                                tool_name=call.name,
                                tool_call_id=call.id,
                                arguments=call.arguments,
                                result=result_payload,
                                agent_round=round_index + 1,
                                stage="portfolio_agent",
                                schema_version=schema_version,
                            )
                        )
                    ctx.add_tool_result_message(call.id, call.name, result_payload)
                continue

            ctx.add_message("assistant", response.text or "")
            if response.text:
                work_state["reasoning_text"] = str(response.text).strip()
            result = self._parse_final_result(response.text or "", work_state=work_state)
            result.messages = list(ctx.messages)
            if "excluded_tickers" not in result.metadata:
                result.metadata["excluded_tickers"] = list(agent_context.excluded_tickers)
            if work_state["tool_invocations"]:
                result.metadata["tool_invocations"] = list(work_state["tool_invocations"])
            if work_state.get("reasoning_text"):
                result.metadata["reasoning_text"] = work_state["reasoning_text"]
            self._record_event(
                "agent_complete",
                agent_context,
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
        status_callback: StepStatusCallback | None,
        massive_client: Any,
        work_state: dict[str, Any],
    ) -> dict[str, Any]:
        schema_version = int(self.config.get("event_store", {}).get("schema_version", 1))
        self.event_store.record(
            EventRecord(
                event_type="tool_call",
                schema_version=schema_version,
                session_id=context.session_id or "n/a",
                run_id=context.run_id or "n/a",
                tool_name=name,
                tool_arguments=arguments,
                agent="portfolio_agent",
            )
        )
        logger.info("Tool invocation: %s args=%s", name, arguments)
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
            work_state["tickers"] = filtered
            return payload

        if name == "fetch_ticker_data":
            payload = fetch_ticker_data_tool(
                arguments,
                tickr_data_manager=self.tickr_data_manager,
                stock_data_fetcher=self._stock_data_fetcher,
                history_period=stocks_cfg.get("history_period", "1y"),
                massive_client=massive_client,
                progress_callback=progress_callback,
            )
            work_state["tickers"] = payload.get("available_tickers", work_state.get("tickers", []))
            return payload

        if name == "build_summary":
            payload = build_summary_tool(
                arguments,
                tickr_data_manager=self.tickr_data_manager,
                tickr_summary_manager=self.tickr_summary_manager,
            )
            work_state["summary"] = payload.get("summary", "")
            work_state["tickers"] = payload.get("tickers", work_state.get("tickers", []))
            return payload

        if name == "allocate_weights":
            if status_callback is not None:
                status_callback("allocate_weights")
            payload = allocate_weights_tool(arguments)
            work_state["weights"] = payload.get("normalized_weights", {})
            work_state["allocation"] = payload.get("allocation", {})
            return payload

        if name == "analyze_portfolio":
            if status_callback is not None:
                status_callback("analyze_portfolio")
            if not work_state.get("summary"):
                tickers_for_summary = [str(t).upper() for t in arguments.get("tickers", work_state.get("tickers", []))]
                if tickers_for_summary:
                    summary_payload = build_summary_tool(
                        {"tickers": tickers_for_summary},
                        tickr_data_manager=self.tickr_data_manager,
                        tickr_summary_manager=self.tickr_summary_manager,
                    )
                    work_state["summary"] = summary_payload.get("summary", "")
                    work_state["tickers"] = summary_payload.get("tickers", tickers_for_summary)
            payload = analyze_portfolio_tool(
                arguments,
                tickr_data_manager=self.tickr_data_manager,
            )
            work_state["analysis"] = payload
            return payload

        return {"error": f"Unknown tool: {name}"}

    def _parse_final_result(self, text: str, *, work_state: dict[str, Any] | None = None) -> AgentResult:
        payload = _extract_json_payload(text)
        work_state = work_state or {}
        tickers = [str(item).upper() for item in payload.get("tickers", []) if str(item).strip()]
        if not tickers:
            tickers = [str(item).upper() for item in work_state.get("tickers", []) if str(item).strip()]
        weights_raw = payload.get("weights", {})
        if not isinstance(weights_raw, dict):
            weights_raw = {}
        if not weights_raw:
            weights_raw = work_state.get("weights", {})
        weights = {str(k).upper(): float(v) for k, v in weights_raw.items()}

        allocation_raw = payload.get("allocation", {})
        if not isinstance(allocation_raw, dict):
            allocation_raw = {}
        if not allocation_raw:
            allocation_raw = work_state.get("allocation", {})
        allocation = {str(k).upper(): float(v) for k, v in allocation_raw.items()}

        suggestions = payload.get("suggestions", {})
        if not isinstance(suggestions, dict):
            suggestions = {}
        if not suggestions:
            changes = payload.get("changes", {})
            if isinstance(changes, dict):
                suggestions = changes
        suggestions = self._ensure_suggestions(suggestions, weights)

        analysis_text = str(payload.get("analysis_text", text)).strip()
        if not analysis_text and work_state.get("analysis"):
            analysis_text = "Portfolio analysis completed using tool outputs."
        data_by_ticker = self.tickr_data_manager.get_data_by_ticker(tickers)
        summary_text = self.tickr_summary_manager.build_or_get_summary(
            tickers=tickers,
            data_by_ticker=data_by_ticker,
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

    @staticmethod
    def _ensure_suggestions(suggestions: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
        add = suggestions.get("add", []) if isinstance(suggestions, dict) else []
        remove = suggestions.get("remove", []) if isinstance(suggestions, dict) else []
        reweight = suggestions.get("reweight", {}) if isinstance(suggestions, dict) else {}
        if add or remove or reweight:
            return {
                "add": add if isinstance(add, list) else [],
                "remove": remove if isinstance(remove, list) else [],
                "reweight": reweight if isinstance(reweight, dict) else {},
            }

        if not weights:
            return {}

        max_ticker = max(weights, key=lambda ticker: float(weights[ticker]))
        max_weight = float(weights[max_ticker])
        if max_weight <= 0.65:
            return {}

        target_max = 0.55
        excess = max_weight - target_max
        others = [ticker for ticker in weights if ticker != max_ticker]
        if not others:
            return {}

        distribute = excess / len(others)
        fallback = {max_ticker: round(target_max, 4)}
        for ticker in others:
            fallback[ticker] = round(float(weights.get(ticker, 0.0)) + distribute, 4)
        return {"add": [], "remove": [], "reweight": fallback}

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

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "portfolio_agent_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "weights": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "allocation": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "analysis_text": {"type": "string"},
                        "suggestions": {
                            "type": "object",
                            "properties": {
                                "add": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "remove": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "reweight": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number"},
                                },
                            },
                            "required": ["add", "remove", "reweight"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tickers", "weights", "allocation", "analysis_text", "suggestions"],
                    "additionalProperties": False,
                },
            },
        }


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
