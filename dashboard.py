"""
LIBB Dashboard — visual overview of all model runs.
Run with:  streamlit run dashboard.py
"""

import json
import glob
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

RUNS_ROOT = Path("user_side/runs/run_v1")

st.set_page_config(page_title="LIBB Dashboard", layout="wide", page_icon="📈")
st.title("📈 LIBB — LLM Investor Behavior Dashboard")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_models():
    if not RUNS_ROOT.exists():
        return []
    return sorted([p.name for p in RUNS_ROOT.iterdir() if p.is_dir()])


def load_latest_log(model: str) -> dict | None:
    log_dir = RUNS_ROOT / model / "logging"
    logs = sorted(log_dir.glob("*.json")) if log_dir.exists() else []
    if not logs:
        return None
    with open(logs[-1]) as f:
        return json.load(f)


def load_portfolio_history(model: str) -> pd.DataFrame:
    path = RUNS_ROOT / model / "portfolio" / "portfolio_history.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_trade_log(model: str) -> pd.DataFrame:
    path = RUNS_ROOT / model / "portfolio" / "trade_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_pending_orders(model: str) -> list:
    path = RUNS_ROOT / model / "portfolio" / "pending_trades.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("orders", [])


def load_sentiment(model: str) -> pd.DataFrame:
    path = RUNS_ROOT / model / "metrics" / "sentiment.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data) if data else pd.DataFrame()


def load_latest_report(model: str) -> str | None:
    for subdir in ["daily_reports", "deep_research"]:
        report_dir = RUNS_ROOT / model / "research" / subdir
        if not report_dir.exists():
            continue
        reports = sorted(report_dir.glob("*.txt"))
        if reports:
            return reports[-1].read_text(encoding="utf-8")
    return None


# ─────────────────────────────────────────────
# Model selector
# ─────────────────────────────────────────────

models = get_models()
if not models:
    st.error("No model runs found. Run the workflow first.")
    st.stop()

selected = st.sidebar.selectbox("Model", models, index=0)
st.sidebar.markdown("---")
st.sidebar.caption(f"Runs root: `{RUNS_ROOT}`")
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# ─────────────────────────────────────────────
# Latest run status
# ─────────────────────────────────────────────

st.subheader("Latest Run")
log = load_latest_log(selected)

if log:
    status_color = {"SUCCESS": "🟢", "SKIPPED": "🟡", "FAILURE": "🔴"}.get(log.get("processing_status", ""), "⚪")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status", f"{status_color} {log.get('processing_status', 'N/A')}")
    col2.metric("Date", log.get("date", "—"))
    col3.metric("Portfolio Value", f"${log.get('portfolio_value', 0):,.2f}")
    col4.metric("Orders Filled", log.get("orders_processed", 0))
    col5.metric("Orders Failed", log.get("orders_failed", 0))

    with st.expander("Full log"):
        st.json(log)
else:
    st.info("No run logs found for this model.")

st.divider()

# ─────────────────────────────────────────────
# Equity curve
# ─────────────────────────────────────────────

st.subheader("Portfolio Equity")
history = load_portfolio_history(selected)

if not history.empty and len(history) > 1:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(history["date"], history["equity"], color="#1f77b4", linewidth=2, label="Equity")
    ax.fill_between(history["date"], history["equity"], alpha=0.1, color="#1f77b4")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    latest = history.iloc[-1]
    col1.metric("Current Equity", f"${latest['equity']:,.2f}")
    col2.metric("Cash", f"${latest['cash']:,.2f}")
    col3.metric("Overall Return", f"{latest['overall_return_pct']:.2f}%",
                delta=f"{latest['daily_return_pct']:.2f}% today" if pd.notna(latest.get('daily_return_pct')) else None)
elif not history.empty:
    st.info(f"Only 1 data point so far ({history.iloc[-1]['date'].date()}). Chart will appear after more runs.")
    st.dataframe(history, use_container_width=True)
else:
    st.info("No portfolio history yet.")

st.divider()

# ─────────────────────────────────────────────
# Sentiment
# ─────────────────────────────────────────────

st.subheader("Sentiment")
sentiment_df = load_sentiment(selected)

if not sentiment_df.empty:
    if len(sentiment_df) > 1:
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

        ax1.plot(sentiment_df["date"], sentiment_df["polarity"], color="green", marker="o")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("Polarity")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        ax1.grid(True, alpha=0.3)

        ax2.plot(sentiment_df["date"], sentiment_df["subjectivity"], color="orange", marker="o")
        ax2.set_title("Subjectivity")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)
    else:
        row = sentiment_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Polarity", f"{row.get('polarity', 0):.3f}")
        c2.metric("Subjectivity", f"{row.get('subjectivity', 0):.3f}")
        c3.metric("Positive words", int(row.get('positive_count', 0)))
        c4.metric("Negative words", int(row.get('negative_count', 0)))
else:
    st.info("No sentiment data yet.")

st.divider()

# ─────────────────────────────────────────────
# Trade log
# ─────────────────────────────────────────────

st.subheader("Trade Log")
trade_log = load_trade_log(selected)

if not trade_log.empty:
    st.dataframe(trade_log.sort_values("date", ascending=False) if "date" in trade_log.columns else trade_log,
                 use_container_width=True, height=250)
else:
    st.info("No trades yet.")

st.divider()

# ─────────────────────────────────────────────
# Pending orders
# ─────────────────────────────────────────────

st.subheader("Pending Orders")
pending = load_pending_orders(selected)

if pending:
    st.dataframe(pd.DataFrame(pending), use_container_width=True)
else:
    st.info("No pending orders.")

st.divider()

# ─────────────────────────────────────────────
# Latest report
# ─────────────────────────────────────────────

st.subheader("Latest Report")
report = load_latest_report(selected)

if report:
    with st.expander("Show full report", expanded=True):
        st.text_area("", report, height=400, label_visibility="collapsed")
else:
    st.info("No reports found.")
