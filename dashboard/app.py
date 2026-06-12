"""
Streamlit Real-Time Dashboard — displays live metrics, fraud alerts,
LLM-generated business insights, and ML prediction distributions.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Data Pipeline — Live Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2530; border-radius: 10px;
        padding: 1rem; margin: 0.4rem 0;
    }
    .alert-critical { color: #ff4b4b; font-weight: bold; }
    .alert-high     { color: #ffa500; }
    .alert-medium   { color: #ffd700; }
    .insight-box {
        background: #0e1117; border-left: 4px solid #00c0f0;
        padding: 1rem; border-radius: 4px; margin: 0.5rem 0;
    }
    .stMetric { background: #1e2530; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data source: Redis → API → mock fallback
# ─────────────────────────────────────────────

@st.cache_resource
def get_data_source():
    """Return a unified data accessor (Redis preferred, HTTP API fallback, mock last)."""
    # Try Redis
    try:
        from backend.storage.storage import RedisStorage
        redis = RedisStorage(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
        )
        if redis._connected:
            return ("redis", redis)
    except Exception:
        pass

    # Try REST API
    api_url = os.getenv("API_URL", "http://localhost:8080")
    try:
        import requests
        resp = requests.get(f"{api_url}/health", timeout=2)
        if resp.ok:
            return ("api", api_url)
    except Exception:
        pass

    return ("mock", None)


def fetch_metrics_history(source, n: int = 60) -> List[Dict[str, Any]]:
    kind, client = source
    if kind == "redis":
        return client.get_metrics_history(n=n)
    if kind == "api":
        try:
            import requests
            r = requests.get(f"{client}/metrics/history?n={n}", timeout=3)
            return r.json().get("data", []) if r.ok else []
        except Exception:
            return []
    # Mock
    return _generate_mock_history(n)


def fetch_latest_metrics(source) -> Dict[str, Any]:
    kind, client = source
    if kind == "redis":
        return client.get_latest_metrics() or {}
    if kind == "api":
        try:
            import requests
            r = requests.get(f"{client}/metrics/latest", timeout=3)
            return r.json().get("data", {}) if r.ok else {}
        except Exception:
            return {}
    return _generate_mock_metrics()


def fetch_latest_insight(source) -> Dict[str, Any]:
    kind, client = source
    if kind == "redis":
        return client.get_latest_insight() or {}
    if kind == "api":
        try:
            import requests
            r = requests.get(f"{client}/insights/latest", timeout=3)
            return r.json().get("data", {}) if r.ok else {}
        except Exception:
            return {}
    return _generate_mock_insight()


def fetch_alerts(source, limit: int = 20) -> List[Dict[str, Any]]:
    kind, client = source
    if kind == "api":
        try:
            import requests
            r = requests.get(f"{client}/alerts?limit={limit}", timeout=3)
            return r.json().get("data", []) if r.ok else []
        except Exception:
            return []
    return _generate_mock_alerts()


# ─────────────────────────────────────────────
# Mock data generators (fallback for offline demo)
# ─────────────────────────────────────────────

import random
import numpy as np

_rng = random.Random(42)


def _generate_mock_history(n: int = 60) -> List[Dict[str, Any]]:
    history = []
    for i in range(n):
        txns   = _rng.randint(50, 400)
        fraud  = _rng.randint(0, int(txns * 0.12))
        history.append({
            "window_start":       f"2024-01-15T{i // 60:02d}:{i % 60:02d}:00",
            "total_transactions": txns,
            "total_amount":       round(txns * _rng.uniform(200, 600), 2),
            "avg_amount":         round(_rng.uniform(150, 700), 2),
            "max_amount":         round(_rng.uniform(1000, 50000), 2),
            "unique_users":       _rng.randint(20, 120),
            "fraud_count":        fraud,
            "fraud_rate":         round(fraud / max(txns, 1), 4),
        })
    return history


def _generate_mock_metrics() -> Dict[str, Any]:
    txns  = _rng.randint(100, 400)
    fraud = _rng.randint(0, 30)
    return {
        "total_transactions": txns,
        "total_amount":       round(txns * _rng.uniform(200, 600), 2),
        "avg_amount":         round(_rng.uniform(150, 700), 2),
        "max_amount":         round(_rng.uniform(1000, 50000), 2),
        "unique_users":       _rng.randint(20, 120),
        "fraud_count":        fraud,
        "fraud_rate":         round(fraud / max(txns, 1), 4),
        "event_type_breakdown": {
            "purchase": _rng.randint(40, 200),
            "transfer": _rng.randint(20, 100),
            "withdrawal": _rng.randint(10, 60),
        },
        "device_breakdown": {
            "mobile": _rng.randint(50, 200),
            "desktop": _rng.randint(30, 150),
            "api": _rng.randint(5, 40),
        },
    }


def _generate_mock_insight() -> Dict[str, Any]:
    return {
        "insight_text": (
            "Transaction volume is within normal range with a slight uptick in mobile payments. "
            "Fraud rate remains below 5%, but 3 high-value API transactions flagged for review. "
            "Night-time activity from unknown locations warrants monitoring."
        ),
        "key_findings": [
            "Average transaction value increased 12% vs prior window",
            "API device fraud rate 3× higher than mobile/desktop",
            "New York and Los Angeles account for 60% of transaction volume",
        ],
        "recommendations": [
            "Increase scrutiny on API transactions above $5,000 at night",
            "Enable step-up authentication for high-value transfers from new devices",
        ],
        "risk_level": "medium",
        "generated_at": datetime.utcnow().isoformat(),
    }


def _generate_mock_alerts() -> List[Dict[str, Any]]:
    severities = ["critical", "high", "medium", "low"]
    types      = ["fraud_detected", "high_transaction", "failed_attempts", "anomaly"]
    return [
        {
            "alert_id":    f"alert-{i:04d}",
            "alert_type":  _rng.choice(types),
            "severity":    _rng.choice(severities),
            "title":       f"Alert #{i} — {_rng.choice(['Fraud Risk', 'High Amount', 'Velocity Spike'])}",
            "description": f"User user_{_rng.randint(1,50):04d} triggered rule at {datetime.utcnow().isoformat()}",
            "user_id":     f"user_{_rng.randint(1, 200):04d}",
            "triggered_at": datetime.utcnow().isoformat(),
        }
        for i in range(1, 16)
    ]


# ─────────────────────────────────────────────
# Dashboard layout
# ─────────────────────────────────────────────

def render_header(source_kind: str) -> None:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("🔍 LLM Data Pipeline — Live Monitor")
        st.caption(f"Data source: **{source_kind.upper()}** | Refreshes every 10 s")
    with col2:
        st.metric("Status", "🟢 Live")


def render_kpi_row(metrics: Dict[str, Any]) -> None:
    st.subheader("📊 Current Window KPIs")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Transactions",  metrics.get("total_transactions", 0))
    c2.metric("Total Volume",        f"${metrics.get('total_amount', 0):,.0f}")
    c3.metric("Avg Transaction",     f"${metrics.get('avg_amount', 0):,.0f}")
    c4.metric("Unique Users",        metrics.get("unique_users", 0))
    fraud_rate = metrics.get("fraud_rate", 0)
    c5.metric("Fraud Count",         metrics.get("fraud_count", 0),
              delta=None, delta_color="inverse")
    c6.metric("Fraud Rate",          f"{fraud_rate:.1%}",
              delta=None, delta_color="inverse")


def render_throughput_chart(history: List[Dict[str, Any]]) -> None:
    if not history:
        st.info("Waiting for data…")
        return
    df = pd.DataFrame(history)
    if "window_start" not in df.columns:
        return
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
    df = df.dropna(subset=["window_start"]).sort_values("window_start")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["window_start"], y=df["total_transactions"],
        name="Transactions", line=dict(color="#00c0f0", width=2), fill="tozeroy",
    ))
    fig.add_trace(go.Scatter(
        x=df["window_start"], y=df.get("fraud_count", pd.Series([0]*len(df))),
        name="Fraud", line=dict(color="#ff4b4b", width=2),
    ))
    fig.update_layout(
        title="Transaction Throughput & Fraud Count Over Time",
        xaxis_title="Time", yaxis_title="Count",
        template="plotly_dark", height=320,
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_fraud_rate_chart(history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    if "fraud_rate" not in df.columns:
        return
    df["window_start"] = pd.to_datetime(df.get("window_start", pd.Series()), errors="coerce")
    df = df.dropna(subset=["window_start"]).sort_values("window_start")
    df["fraud_pct"] = df["fraud_rate"] * 100

    fig = px.area(
        df, x="window_start", y="fraud_pct",
        title="Fraud Rate % Over Time",
        labels={"fraud_pct": "Fraud %", "window_start": "Time"},
        template="plotly_dark", color_discrete_sequence=["#ff6b6b"],
    )
    fig.add_hline(y=5, line_dash="dash", line_color="orange",
                  annotation_text="5% threshold")
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)


def render_breakdown_charts(metrics: Dict[str, Any]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        device = metrics.get("device_breakdown", {})
        if device:
            fig = px.pie(
                names=list(device.keys()), values=list(device.values()),
                title="Device Type Distribution",
                template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        events = metrics.get("event_type_breakdown", {})
        if events:
            fig = px.bar(
                x=list(events.keys()), y=list(events.values()),
                title="Event Type Breakdown",
                labels={"x": "Event Type", "y": "Count"},
                template="plotly_dark",
                color=list(events.values()),
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_insight(insight: Dict[str, Any]) -> None:
    st.subheader("🤖 Latest LLM Business Insight")
    risk = insight.get("risk_level", "low")
    risk_color = {
        "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"
    }.get(risk, "⚪")

    st.markdown(f"""
    <div class="insight-box">
        <p><b>{risk_color} Risk Level: {risk.upper()}</b></p>
        <p>{insight.get("insight_text", "Awaiting insight generation…")}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔍 Key Findings**")
        for finding in insight.get("key_findings", []):
            st.markdown(f"• {finding}")
    with col2:
        st.markdown("**💡 Recommendations**")
        for rec in insight.get("recommendations", []):
            st.markdown(f"• {rec}")


def render_alerts(alerts: List[Dict[str, Any]]) -> None:
    st.subheader("🚨 Recent Alerts")
    if not alerts:
        st.success("No recent alerts — all clear!")
        return

    severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    df = pd.DataFrame(alerts)
    if "severity" in df.columns:
        counts = df["severity"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Critical", counts.get("critical", 0))
        c2.metric("🟠 High",     counts.get("high", 0))
        c3.metric("🟡 Medium",   counts.get("medium", 0))
        c4.metric("🟢 Low",      counts.get("low", 0))

    for alert in alerts[:10]:
        sev  = alert.get("severity", "low")
        icon = severity_icon.get(sev, "⚪")
        with st.expander(f"{icon} {alert.get('title', 'Alert')} — User: {alert.get('user_id', 'N/A')}"):
            st.write(alert.get("description", ""))
            col1, col2 = st.columns(2)
            col1.write(f"**Type:** {alert.get('alert_type', 'N/A')}")
            col2.write(f"**Time:** {alert.get('triggered_at', 'N/A')}")


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────

def main() -> None:
    source = get_data_source()
    source_kind = source[0]

    render_header(source_kind)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        refresh_interval = st.slider("Auto-refresh (seconds)", 5, 60, 10)
        history_window   = st.slider("History window (points)", 10, 200, 60)
        show_raw         = st.checkbox("Show raw metrics JSON", value=False)
        st.divider()
        st.markdown("### Pipeline Quickstart")
        st.code("python -m backend.pipeline pipeline --events 500", language="bash")
        st.markdown("### API Docs")
        st.markdown("[http://localhost:8080/docs](http://localhost:8080/docs)")

    # Fetch data
    metrics = fetch_latest_metrics(source)
    history = fetch_metrics_history(source, n=history_window)
    insight = fetch_latest_insight(source)
    alerts  = fetch_alerts(source)

    # KPIs
    render_kpi_row(metrics)
    st.divider()

    # Charts
    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_throughput_chart(history)
    with col_right:
        render_fraud_rate_chart(history)

    render_breakdown_charts(metrics)
    st.divider()

    # Insight + Alerts
    col_ins, col_alert = st.columns([1, 1])
    with col_ins:
        render_insight(insight)
    with col_alert:
        render_alerts(alerts)

    # Raw JSON
    if show_raw:
        st.divider()
        st.subheader("📦 Raw Metrics")
        st.json(metrics)

    # Auto-refresh
    st.caption(f"⏱ Last updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    time.sleep(refresh_interval)
    st.rerun()


if __name__ == "__main__":
    main()
