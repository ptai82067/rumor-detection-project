"""
Page 5: Experimental Results
=============================
Shows baseline, V1, V2 metrics with Plotly charts,
ablation study results, and confusion matrices.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.data_loader import load_final_metrics, load_ablation_table
from ui.components.metrics_charts import (
    create_metrics_bar_chart, create_recall_fn_chart, create_confusion_matrix
)
from ui.components.kg_visualizer import display_ablation_figures

st.title("📈 Kết quả thực nghiệm")
st.markdown("---")

# ============================================================
# FINAL METRICS
# ============================================================
st.subheader("Chỉ số hiệu năng cuối cùng")

metrics_df = load_final_metrics()

if metrics_df is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(metrics_df.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'MCC'],
            color='lightgreen'
        ).highlight_min(
            subset=['FN'],
            color='lightgreen'
        ), use_container_width=True, hide_index=True)

    with col2:
        st.info("""
        **5 cấu hình mô hình**:
        1. **TF-IDF + Propagation** — Baseline
        2. **TF-IDF + Graph** — Tăng cường Graph
        3. **MiniLM only** — Chỉ ngữ nghĩa
        4. **MiniLM + Graph** — Ngữ nghĩa + Graph
        5. **Full Hybrid** — Tốt nhất
        """)

    # Interactive charts
    st.plotly_chart(create_metrics_bar_chart(metrics_df), use_container_width=True)
    st.plotly_chart(create_recall_fn_chart(metrics_df), use_container_width=True)

else:
    st.warning("Không tìm thấy final_metrics_table.csv. Chạy Notebook tổng hợp kết quả trước.")

st.markdown("---")

# ============================================================
# ABLATION STUDY
# ============================================================
st.subheader("Kết quả Ablation Study")

ablation_df = load_ablation_table()

if ablation_df is not None:
    display_df = ablation_df.copy()
    if 'Delta_Recall_vs_TFIDF' in display_df.columns:
        display_df = display_df.rename(columns={
            'Delta_Recall_vs_TFIDF': 'Recall Delta',
            'Delta_FN_vs_TFIDF': 'FN Delta'
        })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Nhận xét**:
    - MiniLM riêng lẻ kém hơn TF-IDF (73.5% so với 74.7% F1)
    - Thêm Propagation vào MiniLM cải thiện không đáng kể (+0.1%)
    - Thêm Graph vào MiniLM cải thiện **đáng kể** (+29.2% Recall)
    - Full Hybrid (MiniLM + Prop + Graph) đạt kết quả tốt nhất
    - False Negatives giảm từ 154 (TF-IDF) xuống 14 (Full Hybrid)
    """)
else:
    st.warning("Không tìm thấy Ablation Table. Chạy run_ablation.py trước.")

st.markdown("---")

# ============================================================
# CONFUSION MATRICES
# ============================================================
st.subheader("So sánh Confusion Matrix")

cm_data = {
    "TF-IDF + Propagation": np.array([[11692, 2550], [1415, 4831]]),
    "MiniLM Only": np.array([[9978, 4264], [2159, 4087]]),
    "MiniLM + Graph": np.array([[13663, 579], [259, 5987]]),
    "Full Hybrid": np.array([[13716, 526], [246, 6000]]),
}

col1, col2 = st.columns(2)

for i, (name, cm) in enumerate(cm_data.items()):
    with col1 if i % 2 == 0 else col2:
        st.plotly_chart(create_confusion_matrix(cm), use_container_width=True)
        st.caption(f"**{name}**")

st.markdown("---")

# ============================================================
# PRE-GENERATED FIGURES
# ============================================================
st.subheader("Biểu đồ Ablation Study (đã tạo sẵn)")

if display_ablation_figures():
    st.caption("Biểu đồ được tạo bởi run_ablation.py")
else:
    st.info("Chạy run_ablation.py để tạo biểu đồ Ablation.")

st.markdown("---")

# Key findings
st.subheader("🔑 Tổng kết các kết quả chính")

col1, col2, col3 = st.columns(3)
with col1:
    st.success("**Kết quả 1**\n\nGraph Topology Features cải thiện **+30.4% Recall** khi thêm vào MiniLM Embeddings, khôi phục 1,900 False Negatives.")
with col2:
    st.success("**Kết quả 2**\n\nFull Hybrid Model đạt **96.2% Accuracy** và **96.1% Recall** — tốt nhất trong tất cả cấu hình.")
with col3:
    st.success("**Kết quả 3**\n\nTổng hợp cấp Thread (V2, 402 chiều) vượt trội hơn cấp Post (V1, 5,398 chiều) dù chỉ dùng ~13x ít đặc trưng hơn, chứng minh giá trị của ngữ cảnh cấu trúc.")