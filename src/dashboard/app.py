import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pytz

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# --- IMPORT CORE MODULES ---
# Thêm đường dẫn gốc để import được các module trong src/
sys.path.append(str(Path(__file__).resolve().parents[2]))

from storage.redis_client import get_next_draw
from core_model.registry import get_current_version_meta
from core_model.data_prep import load_multi_hot_data
from core_model.inference import predict_next_draw

# --- CONFIG ---
HISTORY_PATH = Path("data/predictions_history.parquet")
METRICS_PATH = Path("data/model_metrics.parquet")
ACTUAL_DATA_PATH = Path("data/multi_hot_matrix.csv")

# Timezone Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

st.set_page_config(
    page_title="Prediction Monitoring",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS TÙY CHỈNH ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        .main .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HELPER FUNCTIONS ---

def format_datetime_vn(dt_val):
    """Chuyển đổi datetime sang giờ Việt Nam: DD/MM/YYYY - HH:MM:SS"""
    if dt_val is None or pd.isna(dt_val):
        return "-"
    if isinstance(dt_val, str):
        try:
            dt_val = pd.to_datetime(dt_val)
        except:
            return dt_val
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=pytz.utc)
    dt_vn = dt_val.astimezone(VN_TZ)
    return dt_vn.strftime("%d/%m/%Y - %H:%M:%S")

def load_actual_results():
    """Load kết quả thực tế từ CSV"""
    if not ACTUAL_DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(ACTUAL_DATA_PATH, sep=";")
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
        cols_num = [c for c in df.columns if c.startswith("n_")]
        df["actual_numbers"] = df[cols_num].values.tolist()
        return df[["draw_id", "date", "actual_numbers"]]
    except Exception as e:
        st.error(f"Lỗi đọc file kết quả thực: {e}")
        return None

# --- HEADER & SYSTEM STATUS ---
registry = get_current_version_meta()
current_version = registry.get("current_version", "Unknown")

col_head_1, col_head_2 = st.columns([3, 1])
with col_head_1:
    st.title("📊 Prediction Dashboard")
with col_head_2:
    st.info(f"**System Version:** {current_version}")

st.markdown("---")

# --- TABS ---
tab_overview, tab_history, tab_models, tab_simulation = st.tabs(
    ["📈 Overview", "🔍 History & Comparison", "🤖 Model Metrics", "🧪 Backtest Simulation"]
)

# ==================================
# 1) OVERVIEW
# ==================================
with tab_overview:
    st.subheader("Next Draw Prediction")
    next_payload = get_next_draw()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        time_display = "-"
        if next_payload and "ts" in next_payload:
            time_display = format_datetime_vn(next_payload["ts"])
        st.metric("Prediction Time (VN)", time_display)
    with c2:
        ver_display = next_payload.get("version", "-") if next_payload else "-"
        st.metric("Model Version", ver_display)
    with c3:
        source_display = next_payload.get("source", "-") if next_payload else "-"
        st.metric("Source", source_display)

    st.markdown("### Predicted Numbers")
    if next_payload and "numbers" in next_payload:
        nums = sorted(next_payload["numbers"])
        cols = st.columns(6)
        for i, n in enumerate(nums):
            cols[i].metric(f"Ball {i+1}", n)
    else:
        st.warning("Chưa có dữ liệu dự đoán kỳ tiếp theo.")

# ==================================
# 2) HISTORY & COMPARISON
# ==================================
with tab_history:
    c_filter, _ = st.columns([1, 3])
    with c_filter:
        limit_option = st.selectbox("Số kỳ quay gần nhất:", [10, 20, 50, 100, "Tất cả"])

    df_actual = load_actual_results()
    df_pred = None
    if HISTORY_PATH.exists():
        df_pred = pd.read_parquet(HISTORY_PATH)

    if df_actual is not None:
        df_view = df_actual.copy()
        df_view["draw_id"] = pd.to_numeric(df_view["draw_id"], errors='coerce')
        
        if df_pred is not None and not df_pred.empty:
            df_pred["draw_id"] = pd.to_numeric(df_pred["draw_id"], errors='coerce')
            df_view = pd.merge(df_view, df_pred[["draw_id", "numbers", "model_version", "event_time"]], on="draw_id", how="left")
            
            def calc_hits(row):
                if isinstance(row["numbers"], (list, np.ndarray)) and isinstance(row["actual_numbers"], (list, np.ndarray)):
                    inter = set(row["numbers"]) & set(row["actual_numbers"])
                    return len(inter), list(inter)
                return 0, []

            res = df_view.apply(calc_hits, axis=1, result_type='expand')
            df_view["Hits"] = res[0]
            df_view["Matched"] = res[1]
            df_view["Predict Time"] = pd.to_datetime(df_view["event_time"]).apply(format_datetime_vn)
        else:
            df_view["numbers"] = None
            df_view["Hits"] = 0
            df_view["Matched"] = None
            df_view["Predict Time"] = "-"
            df_view["model_version"] = "-"

        df_view = df_view.sort_values("draw_id", ascending=False)
        if limit_option != "Tất cả":
            df_view = df_view.head(limit_option)

        final_df = df_view[["draw_id", "date", "actual_numbers", "numbers", "Hits", "Matched", "Predict Time", "model_version"]].copy()
        final_df.columns = ["Draw ID", "Draw Date", "Actual Result", "Prediction", "Hits", "Matched Balls", "Pred Time (VN)", "Model Ver"]
        final_df["Draw Date"] = pd.to_datetime(final_df["Draw Date"]).dt.strftime("%d/%m/%Y")

        def highlight(s):
            return ['background-color: #d1e7dd' if v >= 3 else '' for v in s]

        st.markdown(f"### Kết quả {limit_option if limit_option != 'Tất cả' else ''} kỳ quay gần nhất")
        st.dataframe(final_df.style.apply(highlight, subset=["Hits"]), use_container_width=True, height=600)
    else:
        st.error("Không tìm thấy file dữ liệu gốc (multi_hot_matrix.csv).")

# ==================================
# 3) MODEL METRICS
# ==================================
with tab_models:
    st.subheader("Model Performance (Training Evaluation)")
    if METRICS_PATH.exists():
        df_metrics = pd.read_parquet(METRICS_PATH)
        if not df_metrics.empty:
            df_metrics["evaluated_at"] = pd.to_datetime(df_metrics["evaluated_at"]).apply(format_datetime_vn)
            if {"hit_rate_any", "hit_rate_exact"}.issubset(df_metrics.columns):
                chart_data = df_metrics.melt(id_vars=["version"], value_vars=["hit_rate_any", "hit_rate_exact"], var_name="Metric", value_name="Rate")
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('version', axis=alt.Axis(labelAngle=-45), title="Model Version"),
                    y=alt.Y('Rate', title="Hit Rate"),
                    color='Metric',
                    tooltip=['version', 'Metric', 'Rate']
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_metrics.sort_values("version", ascending=False), use_container_width=True)
    else:
        st.warning("Chưa có file metrics.")

# ==================================
# 4) BACKTEST SIMULATION (NEW)
# ==================================
with tab_simulation:
    st.header("🧪 Backtest Simulation")
    st.markdown("""
    Chức năng này cho phép bạn **chạy thử nghiệm (Backtest)** mô hình hiện tại trên dữ liệu quá khứ.
    Hệ thống sẽ giả lập việc dự đoán cho từng kỳ quay trong quá khứ và so sánh ngay với kết quả thực tế để đánh giá hiệu quả.
    """)

    col_sim_1, col_sim_2 = st.columns([1, 3])
    with col_sim_1:
        n_test = st.number_input("Số kỳ quay muốn kiểm thử (gần nhất):", min_value=5, max_value=200, value=20, step=5)
    
    if st.button("🚀 Chạy Mô Phỏng"):
        if not ACTUAL_DATA_PATH.exists():
             st.error("Không tìm thấy dữ liệu gốc `data/multi_hot_matrix.csv`.")
        else:
            with st.spinner("Đang tải dữ liệu và chạy mô phỏng..."):
                # Load dữ liệu gốc
                df, y_all, _ = load_multi_hot_data(str(ACTUAL_DATA_PATH))
                total_rows = len(df)
                
                if total_rows < n_test + 50:
                    st.warning("Dữ liệu lịch sử quá ngắn để chạy mô phỏng (cần ít nhất 50 kỳ để tạo features).")
                else:
                    results = []
                    progress_text = "Operation in progress. Please wait."
                    my_bar = st.progress(0, text=progress_text)
                    
                    # Chạy loop từ quá khứ đến hiện tại
                    start_idx = total_rows - n_test
                    
                    for i in range(start_idx, total_rows):
                        # Cắt dữ liệu giả lập thời điểm quá khứ (chưa biết kết quả kỳ i)
                        df_slice = df.iloc[:i]
                        y_slice = y_all[:i]
                        
                        # Kết quả thực tế của kỳ i
                        actual_row = df.iloc[i]
                        actual_nums = [int(actual_row[f"n_{k}"]) for k in range(1, 7)]
                        draw_id = actual_row["draw_id"]
                        
                        # Dự đoán
                        try:
                            # Hàm này sẽ build features từ df_slice và gọi model predict
                            pred = predict_next_draw(df_slice, y_slice)
                        except Exception as e:
                            pred = []
                            # st.error(f"Lỗi tại kỳ {draw_id}: {e}")

                        # So khớp
                        hits = len(set(pred) & set(actual_nums))
                        
                        results.append({
                            "Draw ID": draw_id,
                            "Date": actual_row["date"].strftime("%d/%m/%Y") if pd.notna(actual_row["date"]) else "-",
                            "Actual": sorted(actual_nums),
                            "Predicted": sorted(pred),
                            "Hits": hits
                        })
                        
                        # Cập nhật thanh tiến trình
                        prog = (i - start_idx + 1) / n_test
                        my_bar.progress(prog, text=f"Đang dự đoán kỳ {draw_id}...")
                    
                    my_bar.empty()
                    
                    # Tổng hợp kết quả
                    res_df = pd.DataFrame(results).sort_values("Draw ID", ascending=False)
                    
                    # Metrics thống kê nhanh
                    avg_hits = res_df["Hits"].mean()
                    win_rate = (res_df["Hits"] >= 3).mean() * 100
                    max_hits = res_df["Hits"].max()
                    
                    # Hiển thị Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Số kỳ mô phỏng", n_test)
                    m2.metric("Trung bình số trúng", f"{avg_hits:.2f}")
                    m3.metric("Tỷ lệ có giải (>=3 số)", f"{win_rate:.1f}%")
                    m4.metric("Trúng nhiều nhất", f"{max_hits} số")
                    
                    # Hiển thị bảng chi tiết
                    def highlight_sim(s):
                        return ['background-color: #d1e7dd' if v >= 3 else '' for v in s]

                    st.dataframe(
                        res_df.style.apply(highlight_sim, subset=["Hits"]), 
                        use_container_width=True,
                        height=500
                    )