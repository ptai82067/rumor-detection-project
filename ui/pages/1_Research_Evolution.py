"""
Page 1: Research Evolution
==========================
Shows the evolution from Ontology V1 -> KG V1 -> Hybrid V1
-> Ontology V2 -> KG V2 -> Hybrid V2 with cards and metrics.
"""
import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.data_loader import (
    load_dataset_statistics, load_final_metrics, load_v1_metadata, load_v2_metadata
)

st.title("🔬 Ход исследования")
st.markdown("---")

# Project Overview
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ## Обнаружение слухов в соцсетях с использованием Онтологии и Графа знаний
    
    В этом исследовании рассматривается интеграция **признаков Графа знаний**
    с **семантическими эмбеддингами** для улучшения обнаружения слухов в социальных сетях.
    
    **Основной вклад**: Комбинация структурных признаков графа (Centrality, User Reputation,
    Thread Topology) с MiniLM эмбеддингами для гибридной классификации.
    """)

with col2:
    stats = load_dataset_statistics()
    if stats:
        st.metric("Объём данных", f"{stats.get('Number of Tweets', 'N/A')}")
        st.metric("Диалогов", f"{stats.get('Number of Conversation Threads', 'N/A')}")
        st.metric("Пользователей", f"{stats.get('Number of Users', 'N/A')}")

st.markdown("---")

# Evolution Timeline
st.subheader("📊 Дорожная карта исследования")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### Этап 1: Baseline\n\n"
            "**TF-IDF**\n\n"
            "- Logistic Regression\n"
            "- 5,000 TF-IDF признаков\n"
            "- Recall: ~61%")

with col2:
    st.warning("### Этап 2: Интеграция графа\n\n"
               "**Извлечение признаков KG**\n\n"
               "- PHEME Ontology V1/V2\n"
               "- Построение графа ответов\n"
               "- 14 топологических признаков графа\n"
               "- Улучшение Recall: +30.4%")

with col3:
    st.success("### Этап 3: Полный гибрид\n\n"
               "**MiniLM + Graph + Propagation**\n\n"
               "- Sentence Transformer (384 измерения)\n"
               "- Слияние признаков (402 измерения)\n"
               "- Accuracy 98.11%\n"
               "- Rumor Recall 96.46%")

st.markdown("---")

# Pipeline Architecture
st.subheader("🏗️ Полная архитектура конвейера")

st.markdown("""
```
Сырые данные PHEME (5 событий, 5,802 Threads, 103,212 Tweets)
    │
    ├── Конвейер предобработки
    │   ├── Анализ Thread (Source + Reactions)
    │   ├── Нормализация и очистка
    │   └── Извлечение признаков распространения
    │
    ├── Построение Графа знаний
    │   ├── Ontology V1 (Классы: Post, User, Event, Thread)
    │   ├── Ontology V2 (Подклассы: SourcePost, ReplyPost)
    │   ├── Генерация RDF Triple (76,066 nodes, 65,565 edges)
    │   └── Обнаружение и удаление циклов
    │
    ├── Извлечение признаков графа
    │   ├── Центральность узлов (PageRank, Degree, Betweenness)
    │   ├── Репутация пользователя (Rumor Ratio, Post Count)
    │   ├── Структура Thread (Subtree Size, Sibling Count)
    │   └── Авторитет источника (Credibility, Network Size)
    │
    ├── Семантические эмбеддинги (MiniLM)
    │   └── 384-мерные Sentence Embeddings
    │
    └── Гибридный классификатор
        ├── Слияние признаков (402 или 5,398 измерений)
        ├── Logistic Regression
        └── Результат: Слух / Не слух
```
""")

st.markdown("---")

# Performance Summary
st.subheader("📈 Эволюция производительности")

metrics_df = load_final_metrics()
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Базовый Recall", f"{metrics_df['Recall'].iloc[0]:.1%}")
    with col2:
        best_recall = metrics_df['Recall'].max()
        best_model = metrics_df.loc[metrics_df['Recall'].idxmax(), 'Model']
        st.metric("Лучший Recall", f"{best_recall:.1%}", delta=f"+{best_recall - metrics_df['Recall'].iloc[0]:.1%}")
    with col3:
        st.metric("Лучшая Accuracy", f"{metrics_df['Accuracy'].max():.1%}")
    with col4:
        fn_reduction = metrics_df['FN'].iloc[0] - metrics_df['FN'].iloc[-1]
        st.metric("Снижение FN", f"{fn_reduction:,}", delta=f"-{fn_reduction:,}")
else:
    st.info("Запустите конвейер обучения для создания final_metrics_table.csv")

st.markdown("---")
st.caption("Дипломная работа по обнаружению слухов — Университет Нячанга (NTU)")