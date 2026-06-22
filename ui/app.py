#!/usr/bin/env python3
"""
PHEME Rumor Detection — Thesis Demo UI
========================================
Graduation thesis defense demonstration for:

    Rumor Detection on Social Media using Ontology and Knowledge Graph

Launch with:
    streamlit run ui/app.py
"""
import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Обнаружение слухов — Дипломный проект",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🎓 Дипломный проект")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Обнаружение слухов в соцсетях**
*с использованием Онтологии и Графа знаний*
""")
st.sidebar.markdown("---")

# Check model status
v1_meta_path = os.path.join(PROJECT_ROOT, "models", "v1", "metadata.json")
v2_meta_path = os.path.join(PROJECT_ROOT, "models", "v2", "metadata.json")
v1_ready = os.path.exists(v1_meta_path)
v2_ready = os.path.exists(v2_meta_path)

if v1_ready or v2_ready:
    st.sidebar.success("✅ Модели готовы" if (v1_ready and v2_ready) else "⚠️ Модели неполные")
else:
    st.sidebar.warning("⚠️ Сначала запустите скрипт обучения")

st.sidebar.markdown("---")

# Navigation
st.sidebar.markdown("### 📑 Содержание")

st.sidebar.markdown("""
1. 🔬 Ход исследования
2. 🕵️ Обнаружение слухов
3. 🕸️ Изучение Онтологии и KG
4. 📊 Анализ признаков
5. 📈 Результаты экспериментов
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**HCMUT — 2026**")

# ============================================================
# MAIN PAGE — Welcome / Landing
# ============================================================

st.title("🎓 Обнаружение слухов в соцсетях")
st.markdown("## с использованием Онтологии и Графа знаний")
st.markdown("---")

st.markdown("""
### Добро пожаловать в Дипломный проект

Это интерактивное приложение демонстрирует полный процесс исследования по
**обнаружению слухов в социальных сетях** с использованием Графа знаний
и семантических эмбеддингов.

### Разделы для изучения:

| Страница | Описание |
|----------|----------|
| **🔬 Ход исследования** | Дорожная карта от Baseline до Hybrid Model |
| **🕵️ Обнаружение слухов** | Интерактивная демонстрация — выбор диалога и предсказание |
| **🕸️ Изучение Онтологии и KG** | Изучение PHEME Ontology V1/V2 и визуализация Графа знаний |
| **📊 Анализ признаков** | Детальный анализ всех измерений признаков и групп признаков |
| **📈 Результаты экспериментов** | Итоговые показатели, Ablation Study, матрицы ошибок, графики |
""")

st.markdown("---")

# Quick stats dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Модели", "2 (V1 + V2)", help="V1: Post-Level (5,398 измерений), V2: Thread-Level (402 измерения)")
with col2:
    st.metric("Лучшая Accuracy", "98.11%", help="Полная гибридная модель (V2)")
with col3:
    st.metric("Лучший Recall", "96.46%", help="Полная гибридная модель (V2)")
with col4:
    st.metric("Снижение FN", "-1,169", help="По сравнению с TF-IDF Baseline")

st.markdown("---")

with st.expander("ℹ️ Инструкция по использованию"):
    st.markdown("""
    1. **Навигация** с помощью боковой панели или вкладок вверху
    2. **Начните со Страницы 1** для понимания хода исследования
    3. **Попробуйте Страницу 2** для интерактивной демонстрации обнаружения слухов
    4. **Изучите Страницы 3-5** для детального анализа Онтологии, признаков и результатов
    
    **Примечание**: Для демонстрации требуются обученные файлы моделей.
    Выполните следующие команды, если они ещё не созданы:
    
    ```bash
    python scripts/train_and_save_v1.py
    python scripts/train_and_save_v2.py
    ```
    
    Наборы данных, визуализации и таблицы результатов уже загружены
    и не требуют дополнительной обработки.
    """)