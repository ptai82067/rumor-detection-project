"""
KG Visualizer — PHEME Rumor Detection UI
=========================================
Functions for displaying Knowledge Graph visualizations.
"""
import streamlit as st
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def display_static_kg_viz():
    """Display the pre-generated KG visualization image."""
    png_path = os.path.join(PROJECT_ROOT, "visualization", "kg_semantic_final.png")
    svg_path = os.path.join(PROJECT_ROOT, "visualization", "kg_semantic_final.svg")
    
    if os.path.exists(png_path):
        st.image(png_path, caption="Визуализация Графа знаний (Semantic Final)", use_container_width=True)
        return True
    return False


def display_ablation_figures():
    """Display pre-generated ablation study figures."""
    fig_dir = os.path.join(PROJECT_ROOT, "results", "ablation", "figures")
    figures = [
        ("fig1_ablation_bar_metrics.png", "Ablation Study — Сравнение метрик производительности"),
        ("fig2_recall_fn_trend.png", "Тренд Recall и False Negative"),
        ("fig3_confusion_matrices.png", "Матрицы ошибок по конфигурациям"),
        ("fig4_feature_contribution.png", "Анализ вклада признаков"),
    ]
    
    shown = False
    for fname, caption in figures:
        path = os.path.join(fig_dir, fname)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
            shown = True
    return shown


def display_ontology_diagram(version=1):
    """Display ontology diagram if available."""
    # Try to display mermaid diagrams as text
    md_path = os.path.join(PROJECT_ROOT, f"ontology_mermaid_{'class' if version == 1 else 'er'}_diagram.md")
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        st.markdown("### Диаграмма Mermaid")
        st.code(content, language="mermaid")
        st.caption(f"Ontology V{version} — Диаграмма Mermaid")
        return True
    return False