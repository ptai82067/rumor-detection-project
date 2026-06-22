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

st.title("📈 Результаты экспериментов")
st.markdown("---")

# ============================================================
# FINAL METRICS
# ============================================================
st.subheader("Итоговые показатели производительности")

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
        **5 конфигураций модели**:
        1. **TF-IDF + Propagation** — Baseline
        2. **TF-IDF + Graph** — С графом
        3. **MiniLM only** — Только семантика
        4. **MiniLM + Graph** — Семантика + Граф
        5. **Full Hybrid** — Лучшая
        """)

    # Interactive charts
    st.plotly_chart(create_metrics_bar_chart(metrics_df), use_container_width=True)
    st.plotly_chart(create_recall_fn_chart(metrics_df), use_container_width=True)

else:
    st.warning("final_metrics_table.csv не найден. Сначала запустите ноутбук для агрегации результатов.")

st.markdown("---")

# ============================================================
# ABLATION STUDY
# ============================================================
st.subheader("Результаты Ablation Study")

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
    **Наблюдения**:
    - MiniLM отдельно уступает TF-IDF (73.5% против 74.7% F1)
    - Добавление Propagation к MiniLM улучшает незначительно (+0.1%)
    - Добавление Graph к MiniLM улучшает **значительно** (+29.2% Recall)
    - Full Hybrid (MiniLM + Prop + Graph) показывает лучший результат
    - False Negatives снизились с 154 (TF-IDF) до 14 (Full Hybrid)
    """)
else:
    st.warning("Таблица Ablation не найдена. Сначала запустите run_ablation.py.")

st.markdown("---")

# ============================================================
# CONFUSION MATRICES
# ============================================================
st.subheader("Сравнение матриц ошибок")

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
st.subheader("Графики Ablation Study (предварительно созданные)")

if display_ablation_figures():
    st.caption("Графики созданы с помощью run_ablation.py")
else:
    st.info("Запустите run_ablation.py для создания графиков Ablation.")

st.markdown("---")

# Key findings
st.subheader("🔑 Итоги основных результатов")

col1, col2, col3 = st.columns(3)
with col1:
    st.success("**Результат 1**\n\nТопологические признаки графа улучшают **+30.4% Recall** при добавлении к MiniLM Embeddings, восстанавливая 1,900 False Negatives.")
with col2:
    st.success("**Результат 2**\n\nПолная гибридная модель достигает **96.2% Accuracy** и **96.1% Recall** — лучший показатель среди всех конфигураций.")
with col3:
    st.success("**Результат 3**\n\nАгрегация на уровне Thread (V2, 402 измерения) превосходит Post-level (V1, 5,398 измерений), используя ~13x меньше признаков, что доказывает ценность структурного контекста.")