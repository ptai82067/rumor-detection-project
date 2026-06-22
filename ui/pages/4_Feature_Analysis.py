"""
Page 4: Feature Analysis
=========================
Shows V1 and V2 feature composition with dimension comparison charts,
feature group descriptions, and detailed breakdown tables.
"""
import streamlit as st
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.metrics_charts import create_feature_dimension_chart

st.title("📊 Анализ признаков")
st.markdown("---")

# Feature Dimension Overview
st.subheader("Сравнение размерностей признаков")

col1, col2 = st.columns(2)

with col1:
    st.info("### V1 — Post-Level (5,398 измерений)")
    st.markdown("""
    | Компонент | Размерность | Источник |
    |-----------|-----------|--------|
    | TF-IDF | **5,000** | `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` |
    | MiniLM | **384** | `sentence-transformers/all-MiniLM-L6-v2` |
    | Graph | **14** | Post-level: Centrality, Reputation, Thread Structure, Authority |
    | **Всего** | **5,398** | `[TF-IDF + MiniLM + Graph]` |
    """)

with col2:
    st.success("### V2 — Thread-Level (402 измерения)")
    st.markdown("""
    | Компонент | Размерность | Источник |
    |-----------|-----------|--------|
    | MiniLM | **384** | `sentence-transformers/all-MiniLM-L6-v2` (исходный текст) |
    | Propagation | **4** | `thread_size, max_depth, avg_depth, reply_rate` |
    | Graph | **14** | Thread-level: Topology, PageRank, Centrality, User Ratio |
    | **Всего** | **402** | `[MiniLM + Propagation + Graph]` |
    """)

# Feature dimension chart
st.plotly_chart(create_feature_dimension_chart(), use_container_width=True)

st.markdown("---")

# V2 Feature Groups
st.subheader("Группы признаков V2 — Детальный анализ")

v2_tab1, v2_tab2, v2_tab3 = st.tabs(["📐 Graph Topology", "📈 Propagation", "🧠 MiniLM"])

with v2_tab1:
    st.markdown("### 14 Thread-Level Graph Features")
    st.markdown("""
    Эти признаки извлекаются из деревьев ответов Графа знаний и отражают
    структурные свойства диалогов.
    """)

    graph_features_data = {
        'Признак': [
            'thread_depth', 'num_nodes', 'num_edges',
            'avg_branching_factor', 'max_branching_factor',
            'source_reply_count', 'leaf_ratio', 'avg_depth',
            'source_pagerank', 'avg_pagerank',
            'source_centrality', 'avg_centrality',
            'user_rumor_ratio', 'unique_users'
        ],
        'Группа': [
            'Topology', 'Topology', 'Topology',
            'Topology', 'Topology',
            'Topology', 'Topology', 'Topology',
            'PageRank', 'PageRank',
            'Centrality', 'Centrality',
            'User', 'User'
        ],
        'Описание': [
            'Максимальная глубина Thread',
            'Общее количество сообщений в Thread',
            'Общее количество отношений ответов',
            'Среднее количество ответов на узел',
            'Максимальное количество ответов на одно сообщение',
            'Количество прямых ответов на исходное сообщение',
            'Доля сообщений без ответов (листовых)',
            'Средняя глубина сообщений',
            'PageRank исходного сообщения',
            'Средний PageRank в Thread',
            'Degree Centrality исходного сообщения',
            'Средняя Degree Centrality в Thread',
            'Доля слухов в истории пользователя',
            'Количество уникальных пользователей в Thread'
        ]
    }
    gf_df = pd.DataFrame(graph_features_data)
    st.dataframe(gf_df, use_container_width=True, hide_index=True)

with v2_tab2:
    st.markdown("### 4 Propagation Features")
    st.markdown("""
    Propagation Features отражают то, как информация распространяется через структуру Thread,
    включая размер, глубину и скорость отклика.
    """)

    prop_features_data = {
        'Признак': ['thread_size', 'max_depth', 'avg_depth', 'reply_rate'],
        'Описание': [
            'Общее количество сообщений в Thread',
            'Максимальная глубина дерева ответов',
            'Средняя глубина всех сообщений',
            'Всего сообщений / (thread_size + 1) — нормированная плотность ответов'
        ]
    }
    pf_df = pd.DataFrame(prop_features_data)
    st.dataframe(pf_df, use_container_width=True, hide_index=True)

with v2_tab3:
    st.markdown("### MiniML Sentence Embeddings")
    st.markdown("""
    **Модель**: `all-MiniLM-L6-v2` от SentenceTransformers

    | Свойство | Значение |
    |----------|-------|
    | Размерность | 384 |
    | Источник | V2: исходный текст (первое сообщение); V1: весь текст |
    | Размер модели | ~80 MB |
    | Тип | Pooled Sentence Embeddings (Mean Pooling) |
    | Обучающие данные | 1B+ пар предложений (контрастивное обучение) |

    MiniLM — это дистиллированная версия BERT, оптимизированная для семантического сходства на уровне предложений.
    Она создаёт эмбеддинги фиксированного размера, отражающие семантическое значение текста.
    """)

st.markdown("---")

# V1 Feature Groups
st.subheader("Группы признаков V1 — Детальный анализ")

v1_tab1, v1_tab2, v1_tab3 = st.tabs(["📝 TF-IDF", "🧠 MiniLM", "🕸️ Graph"])

with v1_tab1:
    st.markdown("### TF-IDF Features (5,000 измерений)")
    st.markdown("""
    | Параметр | Значение |
    |-----------|-------|
    | max_features | 5,000 |
    | ngram_range | (1, 2) |
    | stop_words | 'english' |
    | min_df | 2 |
    | max_df | 0.95 |

    TF-IDF фиксирует лексические шаблоны: униграммы и биграммы, взвешенные
    по обратной частоте документа во всём корпусе.
    """)

with v1_tab2:
    st.markdown("### MiniLM Sentence Embeddings (384 измерения)")
    st.markdown("""
    Та же архитектура, что и в V2, но применяется ко **всему тексту сообщения**,
    а не только к исходному тексту.
    """)

with v1_tab3:
    st.markdown("### 14 Post-Level Graph Features")
    st.markdown("""
    В отличие от агрегации на уровне Thread в V2, Graph Features в V1 вычисляются для каждого сообщения:

    | Группа признаков | Признаки |
    |--------------|----------|
    | **Центральность узла** | `node_in_degree`, `node_out_degree`, `pagerank_score`, `betweenness_centrality`, `closeness_centrality` |
    | **Репутация пользователя** | `user_prior_rumor_ratio`, `user_post_count`, `user_thread_count`, `user_avg_depth` |
    | **Структура Thread** | `subtree_reply_count`, `sibling_count`, `position_in_thread` |
    | **Авторитет источника** | `source_user_credibility`, `source_network_size` |

    **Источник**: `utils/graph_features.py` — вычисляется через `extract_all_graph_features()`
    из графа ответов (NetworkX DiGraph), построенного на основе Графа знаний.
    """)

st.markdown("---")

# Key Insight
st.info("""
**Ключевой вывод**: V2 достигает более высокой производительности (98.11%) всего с 402 признаками
благодаря работе на уровне Thread, в то время как V1 (96.23%) использует 5,398 признаков на уровне Post.
Агрегация на уровне Thread обеспечивает более сильный сигнал путём объединения
контекста диалога в единый вектор признаков.
""")