# src/dashboard/app.py

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from storage.redis_client import get_next_draw, get_redis
from core_model.registry import get_current_version_meta

HISTORY_PATH = Path("data/predictions_history.parquet")
METRICS_PATH = Path("data/model_metrics.parquet")


st.set_page_config(
    page_title="Prediction Monitoring",
    layout="wide",
)

st.title("Prediction Monitoring Dashboard")

redis_client = get_redis()
registry = get_current_version_meta()
current_version = registry.get("current_version")


# =========================
# TABS
# =========================
tab_overview, tab_history, tab_models = st.tabs(
    ["Overview", "Prediction History", "Model Compare"]
)

# ==================================
# 1) OVERVIEW
# ==================================
with tab_overview:
    col1, col2, col3 = st.columns(3)

    # Tổng số record trong lịch sử
    total_preds = 0
    if HISTORY_PATH.exists():
        total_preds = len(pd.read_parquet(HISTORY_PATH))

    with col1:
        st.metric("Total Predictions Logged", total_preds)

    # Last next_draw from Redis
    next_payload = get_next_draw()
    last_time = "-"
    last_value = "[]"
    if next_payload:
        last_time = next_payload.get("ts", "-")
        last_value = next_payload.get("numbers", [])

    with col2:
        st.metric("Last Next-Draw Prediction Time (UTC)", last_time)

    with col3:
        st.metric("Last Next-Draw Prediction", str(last_value))

    st.markdown("---")
    st.subheader("Raw Redis metrics (per minute) [demo placeholder]")
    st.info("Hiện tại chưa có metrics per-minute chi tiết – bạn có thể thêm counter vào Redis nếu muốn.")


# ==================================
# 2) PREDICTION HISTORY
# ==================================
with tab_history:
    st.subheader("Prediction History (from Parquet)")

    if not HISTORY_PATH.exists():
        st.warning("Chưa có file data/predictions_history.parquet. Hãy để hệ thống chạy 1 thời gian.")
    else:
        df_hist = pd.read_parquet(HISTORY_PATH)

        # format
        if "event_time" in df_hist.columns:
            df_hist["event_time"] = pd.to_datetime(df_hist["event_time"])

        # filter UI
        col_from, col_to, col_ver = st.columns(3)

        with col_from:
            start_date = st.date_input(
                "From date",
                value=df_hist["event_time"].min().date()
                if "event_time" in df_hist.columns
                else datetime.utcnow().date(),
            )
        with col_to:
            end_date = st.date_input(
                "To date",
                value=df_hist["event_time"].max().date()
                if "event_time" in df_hist.columns
                else datetime.utcnow().date(),
            )
        with col_ver:
            versions = sorted(df_hist["model_version"].dropna().unique())
            ver_filter = st.selectbox(
                "Model version",
                options=["(all)"] + versions,
                index=0,
            )

        mask = (df_hist["event_time"].dt.date >= start_date) & (
            df_hist["event_time"].dt.date <= end_date
        )
        if ver_filter != "(all)":
            mask &= df_hist["model_version"] == ver_filter

        df_view = df_hist[mask].sort_values("event_time", ascending=False)

        st.dataframe(
            df_view,
            use_container_width=True,
            height=500,
        )

        st.caption("Nguồn: data/predictions_history.parquet")


# ==================================
# 3) MODEL COMPARE
# ==================================
with tab_models:
    st.subheader("Model Version Comparison")

    if not METRICS_PATH.exists():
        st.warning(
            "Chưa có file data/model_metrics.parquet. "
            "Hãy chạy script utils.evaluate_models hoặc để DAG chạy xong 1 lượt training."
        )
    else:
        df_metrics = pd.read_parquet(METRICS_PATH)
        if "evaluated_at" in df_metrics.columns:
            df_metrics["evaluated_at"] = pd.to_datetime(df_metrics["evaluated_at"])

        st.write("**Bảng metrics cho từng version:**")
        st.dataframe(
            df_metrics.sort_values("evaluated_at", ascending=False),
            use_container_width=True,
            height=400,
        )

        if {"hit_rate_any", "hit_rate_exact"}.issubset(df_metrics.columns):
            st.write("**So sánh tỉ lệ trúng số (bar chart):**")
            chart_df = df_metrics.set_index("version")[
                ["hit_rate_any", "hit_rate_exact"]
            ]
            st.bar_chart(chart_df)

        st.markdown(f"**Current serving version:** `{current_version}`")
