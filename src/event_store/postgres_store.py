"""Postgres-backed adapter over the shared agent_monitoring package."""

from __future__ import annotations

import os

from agent_monitoring.models import (
    AgentPerformanceRecord as SharedAgentPerformanceRecord,
    EventRecord as SharedEventRecord,
    LLMCallRecord as SharedLLMCallRecord,
    ToolCallRecord as SharedToolCallRecord,
)
from agent_monitoring.store.postgres import PostgresEventStore as SharedPostgresEventStore

from src.event_store.models import AgentPerformanceRecord, EventRecord, LLMCallRecord, ToolCallRecord


class PostgresEventStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._store: SharedPostgresEventStore | None = None

    def _shared_store(self) -> SharedPostgresEventStore:
        if self._store is None:
            self._store = SharedPostgresEventStore(
                self._dsn,
                app="portfolio-builder-agent",
                service=os.getenv("SERVICE_NAME", "app"),
                environment=os.getenv("ENVIRONMENT", "local"),
                initialize_schema=True,
            )
        return self._store

    def record(self, event: EventRecord) -> None:
        self._shared_store().record(
            SharedEventRecord(
                id=event.event_id,
                app="portfolio-builder-agent",
                service=os.getenv("SERVICE_NAME", "app"),
                environment=os.getenv("ENVIRONMENT", "local"),
                event_type=event.event_type,
                session_id=event.session_id,
                run_id=event.run_id,
                payload=None,
                request_name=event.request_name,
                model=event.model,
                temperature=event.temperature,
                max_tokens=event.max_tokens,
                messages=event.messages,
                raw_output=event.raw_output,
                status_code=event.status_code,
                latency_ms=event.latency_ms,
                token_usage=event.token_usage,
                parsed_output=event.parsed_output,
                validation_errors=event.validation_errors,
                action=event.action,
                action_payload=event.action_payload,
                tool_name=event.tool_name,
                tool_arguments=event.tool_arguments,
                tool_result=event.tool_result,
                tool_call_id=event.tool_call_id,
                agent=event.agent,
                iteration=event.iteration,
                agent_round=event.agent_round,
                schema_version=event.schema_version,
                timestamp=event.timestamp,
            )
        )

    def query(
        self,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        rows = self._shared_store().query(
            app="portfolio-builder-agent",
            session_id=session_id,
            event_type=event_type,
            since=since,
            limit=limit,
        )
        return [
            EventRecord(
                event_id=row.id,
                schema_version=row.schema_version,
                timestamp=row.timestamp,
                session_id=row.session_id,
                run_id=row.run_id,
                event_type=row.event_type,
                request_name=row.request_name,
                model=row.model,
                temperature=row.temperature,
                max_tokens=row.max_tokens,
                messages=row.messages,
                raw_output=row.raw_output,
                status_code=row.status_code,
                latency_ms=row.latency_ms,
                token_usage=row.token_usage,
                parsed_output=row.parsed_output,
                validation_errors=row.validation_errors,
                action=row.action,
                action_payload=row.action_payload,
                tool_name=row.tool_name,
                tool_arguments=row.tool_arguments,
                tool_result=row.tool_result,
                tool_call_id=row.tool_call_id,
                agent=row.agent,
                iteration=row.iteration,
                agent_round=row.agent_round,
            )
            for row in rows
        ]

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self._shared_store().record_llm_call(
            SharedLLMCallRecord(
                id=record.id,
                app="portfolio-builder-agent",
                service=os.getenv("SERVICE_NAME", "app"),
                environment=os.getenv("ENVIRONMENT", "local"),
                session_id=record.session_id,
                run_id=record.run_id,
                timestamp=record.timestamp,
                model=record.model,
                prompt=record.prompt,
                output=record.output,
                output_code=record.output_code,
                latency_ms=record.latency_ms,
                token_usage=record.token_usage,
                temperature=record.temperature,
                max_tokens=record.max_tokens,
                stage=record.stage,
                schema_version=record.schema_version,
            )
        )

    def query_llm_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[LLMCallRecord]:
        rows = self._shared_store().query_llm_calls(
            app="portfolio-builder-agent",
            session_id=session_id,
            run_id=run_id,
            limit=limit,
        )
        return [
            LLMCallRecord(
                id=row.id,
                session_id=row.session_id,
                run_id=row.run_id,
                timestamp=row.timestamp,
                model=row.model,
                prompt=row.prompt,
                output=row.output,
                output_code=row.output_code,
                latency_ms=row.latency_ms,
                token_usage=row.token_usage,
                temperature=row.temperature,
                max_tokens=row.max_tokens,
                stage=row.stage,
                schema_version=row.schema_version,
            )
            for row in rows
        ]

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self._shared_store().record_tool_call(
            SharedToolCallRecord(
                id=record.id,
                app="portfolio-builder-agent",
                service=os.getenv("SERVICE_NAME", "app"),
                environment=os.getenv("ENVIRONMENT", "local"),
                session_id=record.session_id,
                run_id=record.run_id,
                timestamp=record.timestamp,
                tool_name=record.tool_name,
                tool_call_id=record.tool_call_id,
                arguments=record.arguments,
                result=record.result,
                agent_round=record.agent_round,
                stage=record.stage,
                schema_version=record.schema_version,
            )
        )

    def query_tool_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolCallRecord]:
        rows = self._shared_store().query_tool_calls(
            app="portfolio-builder-agent",
            session_id=session_id,
            run_id=run_id,
            limit=limit,
        )
        return [
            ToolCallRecord(
                id=row.id,
                session_id=row.session_id,
                run_id=row.run_id,
                timestamp=row.timestamp,
                tool_name=row.tool_name,
                tool_call_id=row.tool_call_id,
                arguments=row.arguments,
                result=row.result,
                agent_round=row.agent_round,
                stage=row.stage,
                schema_version=row.schema_version,
            )
            for row in rows
        ]

    def record_agent_performance(self, record: AgentPerformanceRecord) -> None:
        self._shared_store().record_agent_performance(
            SharedAgentPerformanceRecord(
                id=record.id,
                app="portfolio-builder-agent",
                service=os.getenv("SERVICE_NAME", "app"),
                environment=os.getenv("ENVIRONMENT", "local"),
                session_id=record.session_id,
                run_id=record.run_id,
                timestamp=record.timestamp,
                model=record.model,
                total_llm_calls=record.total_llm_calls,
                total_tool_calls=record.total_tool_calls,
                total_iterations=record.total_iterations,
                total_latency_ms=record.total_latency_ms,
                total_tokens=record.total_tokens,
                portfolio_return_1y=record.portfolio_return_1y,
                portfolio_current=record.portfolio_current,
                portfolio_min=record.portfolio_min,
                portfolio_max=record.portfolio_max,
                tickers=record.tickers,
                weights=record.weights,
                status=record.status,
                error_message=record.error_message,
                schema_version=record.schema_version,
            )
        )

    def query_agent_performance(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentPerformanceRecord]:
        rows = self._shared_store().query_agent_performance(
            app="portfolio-builder-agent",
            session_id=session_id,
            run_id=run_id,
            limit=limit,
        )
        return [
            AgentPerformanceRecord(
                id=row.id,
                session_id=row.session_id,
                run_id=row.run_id,
                timestamp=row.timestamp,
                model=row.model,
                total_llm_calls=row.total_llm_calls,
                total_tool_calls=row.total_tool_calls,
                total_iterations=row.total_iterations,
                total_latency_ms=row.total_latency_ms,
                total_tokens=row.total_tokens,
                portfolio_return_1y=row.portfolio_return_1y,
                portfolio_current=row.portfolio_current,
                portfolio_min=row.portfolio_min,
                portfolio_max=row.portfolio_max,
                tickers=row.tickers,
                weights=row.weights,
                status=row.status,
                error_message=row.error_message,
                schema_version=row.schema_version,
            )
            for row in rows
        ]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
