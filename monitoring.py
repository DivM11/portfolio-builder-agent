"""Streamlit monitoring dashboard for the Portfolio Builder Agent.

Displays the four SQLite monitoring tables as interactive dataframes:
  - events            Legacy EventRecord log
  - llm_calls         One row per LLM HTTP round-trip
  - tool_calls        One row per agent tool invocation
  - agent_performance ETL-materialised summary per run

Run via docker-compose (monitor-ui service, port 8502) or directly:
    streamlit run monitoring.py

Environment variables:
    DB_PATH          Path to SQLite database (default: data/events.db)
    MONITOR_API_URL  Base URL of the monitoring REST API (default: http://localhost:8000)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Agent Monitoring",
    page_icon="📊",
    layout="wide",
)

_DB_PATH = Path(os.environ.get("DB_PATH", "data/events.db"))
_API_URL = os.environ.get("MONITOR_API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Data helpers (direct SQLite reads — no inter-service HTTP dependency)
# ---------------------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


@st.cache_data(ttl=15)
def _load(table: str, session_id: str, run_id: str, limit: int) -> pd.DataFrame:
    """Read *table* from the SQLite DB with optional session/run filters."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    try:
        cols = _columns(conn, table)
        clauses: list[str] = []
        params: list[object] = []
        if "session_id" in cols and session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if "run_id" in cols and run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        return pd.read_sql_query(
            f"SELECT * FROM {table} {where} ORDER BY timestamp DESC LIMIT ?",
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("📊 Agent Monitoring Dashboard")

with st.sidebar:
    st.header("Filters")
    session_filter = st.text_input("Session ID", value="", placeholder="all sessions")
    run_filter = st.text_input("Run ID", value="", placeholder="all runs")
    row_limit = st.slider("Row limit", min_value=25, max_value=2000, value=200, step=25)

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("**REST API**")
    st.markdown(f"[Interactive docs ↗]({_API_URL}/docs)")
    st.code(f"curl {_API_URL}/agent-performance", language="bash")

# ---------------------------------------------------------------------------
# Guard — DB must exist before we try to query anything
# ---------------------------------------------------------------------------

if not _DB_PATH.exists():
    st.warning(
        f"No database found at **`{_DB_PATH}`**.  "
        "Run the main portfolio app first — it creates the database automatically."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_events, tab_llm, tab_tools, tab_perf = st.tabs(
    ["🗃️ Events", "🤖 LLM Calls", "🔧 Tool Calls", "📈 Agent Performance"]
)

# ---- Events ----------------------------------------------------------------
with tab_events:
    df = _load("events", session_filter, run_filter, row_limit)
    st.caption(f"{len(df):,} rows · `events` table")
    st.dataframe(df, use_container_width=True)

# ---- LLM Calls -------------------------------------------------------------
with tab_llm:
    df = _load("llm_calls", session_filter, run_filter, row_limit)
    st.caption(f"{len(df):,} rows · `llm_calls` table")

    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total calls", f"{len(df):,}")
        if "latency_ms" in df.columns:
            c2.metric("Avg latency", f"{df['latency_ms'].mean():.0f} ms")
        if "output_code" in df.columns:
            c3.metric("HTTP 200 ✓", int((df["output_code"] == 200).sum()))
        if "token_usage" not in df.columns or df["token_usage"].isna().all():
            pass
        st.divider()

    st.dataframe(df, use_container_width=True)

# ---- Tool Calls ------------------------------------------------------------
with tab_tools:
    df = _load("tool_calls", session_filter, run_filter, row_limit)
    st.caption(f"{len(df):,} rows · `tool_calls` table")

    if not df.empty and "tool_name" in df.columns:
        counts = df["tool_name"].value_counts().rename_axis("tool").reset_index(name="calls")
        st.bar_chart(counts.set_index("tool"), height=200)
        st.divider()

    st.dataframe(df, use_container_width=True)

# ---- Agent Performance -----------------------------------------------------
with tab_perf:
    df = _load("agent_performance", session_filter, run_filter, row_limit)
    st.caption(f"{len(df):,} rows · `agent_performance` table")

    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total runs", f"{len(df):,}")
        if "total_llm_calls" in df.columns:
            c2.metric("Avg LLM calls", f"{df['total_llm_calls'].mean():.1f}")
        if "total_latency_ms" in df.columns:
            c3.metric("Avg latency", f"{df['total_latency_ms'].mean():.0f} ms")
        if "total_tokens" in df.columns:
            c4.metric("Avg tokens", f"{df['total_tokens'].mean():.0f}")

        if "status" in df.columns:
            st.divider()
            status_counts = df["status"].value_counts().rename_axis("status").reset_index(name="runs")
            st.bar_chart(status_counts.set_index("status"), height=180)

        st.divider()

    st.dataframe(df, use_container_width=True)
