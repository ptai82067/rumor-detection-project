"""
Page 3: Ontology & KG Explorer
================================
Tabs for Ontology V1, Ontology V2, KG V1, KG V2 visualizations.
Displays existing figures and pre-generated diagrams.
"""
import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.data_loader import load_ontology_text
from ui.components.kg_visualizer import display_static_kg_viz, display_ontology_diagram

st.title("🕸️ Изучение Онтологии и Графа знаний")
st.markdown("---")

# Dataset Statistics Dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Всего Triple (KG v1)", "~76,066 nodes, 65,565 edges")
with col2:
    st.metric("Классов (V1)", "5", help="Event, Post, User, ConversationThread, VeracityLabel")
with col3:
    st.metric("Классов (V2)", "7", help="+ SourcePost, ReplyPost — подклассы")
with col4:
    st.metric("Object Properties", "7+", help="postedBy, aboutEvent, repliesTo, inThread, hasVeracity, ...")

tabs = st.tabs(["Ontology V1", "Ontology V2", "KG V1 Визуализация", "KG V2 Визуализация", "Статистика KG"])

# ============================================================
# TAB 1: Ontology V1
# ============================================================
with tabs[0]:
    st.subheader("PHEME Ontology V1 — Минимальная онтология")
    st.markdown("""
    Ontology V1 определяет 5 основных классов для моделирования распространения слухов в социальных сетях:

    | Класс | Описание |
    |-------|---------|
    | **Event** | Реальное событие, порождающее обсуждение в соцсетях |
    | **Post** | Сообщение в соцсети (твит) из набора PHEME |
    | **User** | Пользователь, создавший сообщение |
    | **ConversationThread** | Диалог, состоящий из исходного сообщения и ответов |
    | **VeracityLabel** | Метка достоверности: слух (1) или не слух (0) |

    **Object Properties**: `postedBy`, `aboutEvent`, `repliesTo`, `inThread`, `hasVeracity`
    """)

    col1, col2 = st.columns(2)
    with col1:
        display_ontology_diagram(version=1)

    with col2:
        text = load_ontology_text(version=1)
        if text:
            with st.expander("Просмотр исходного кода Ontology V1 (TTL)"):
                st.code(text[:2000], language="turtle")

# ============================================================
# TAB 2: Ontology V2
# ============================================================
with tabs[1]:
    st.subheader("PHEME Ontology V2 — Расширенная онтология")
    st.markdown("""
    V2 расширяет V1 дополнительными подклассами и свойствами для более богатого семантического моделирования:

    **Новые подклассы**:

    | Класс | Родитель | Описание |
    |-------|---------|---------|
    | **SourcePost** | Post | Исходное сообщение, начинающее диалог |
    | **ReplyPost** | Post | Сообщение, отвечающее на другое сообщение в диалоге |

    **Новые Data Properties**: `text`, `createdAt`, `depth`, `childrenCount`,
    `timeSinceSource`, `threadSize`, `maxDepth`, `replySpeed`
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("**Основные улучшения V2**")
        st.markdown("""
        - Добавлено различие SourcePost/ReplyPost для анализа структуры Thread
        - Обход дерева ответов: `post → repliesTo → parent`
        - Propagation Metrics: глубина, количество потомков, временные характеристики
        - Агрегация на уровне Thread: размер, глубина, скорость ответов
        - Исправление URI Fragment для извлечения 65,565 Reply Edges
        """)

    with col2:
        text = load_ontology_text(version=2)
        if text:
            with st.expander("Просмотр исходного кода Ontology V2 (TTL)"):
                st.code(text[:2000], language="turtle")

# ============================================================
# TAB 3: KG V1 Visualization
# ============================================================
with tabs[2]:
    st.subheader("Knowledge Graph V1 — Граф ответов на уровне Post")
    st.markdown("""
    KG V1 построен из набора данных PHEME с использованием Ontology V1.
    Он моделирует сообщения, пользователей, события и диалоги вместе с отношениями ответов.
    """)

    col1, col2 = st.columns([3, 1])
    with col1:
        displayed = display_static_kg_viz()
        if not displayed:
            st.info("Изображение KG не найдено. Сначала запустите скрипт визуализации KG.")

    with col2:
        st.markdown("**Статистика KG V1**")
        st.markdown("""
        - **Nodes**: 76,066 (сообщения + пользователи + события + threads)
        - **Edges**: 65,565 отношений ответов
        - **Структура**: Directed Acyclic Graph (дерево для каждого Thread)
        - **Обнаружение циклов**: DFS-based с удалением рёбер
        """)

        st.markdown("**Data Properties**")
        st.markdown("""
        - `text`: Содержимое сообщения
        - `createdAt`: Время создания
        - `depth`: Позиция в Thread
        - `childrenCount`: Количество прямых ответов
        - `timeSinceSource`: Временной интервал
        """)

# ============================================================
# TAB 4: KG V2 Visualization
# ============================================================
with tabs[3]:
    st.subheader("Knowledge Graph V2 — Агрегированный граф на уровне Thread")
    st.markdown("""
    KG V2 расширяет V1 признаками графа на уровне Thread (14 топологических признаков графа)
    для улучшения обнаружения слухов. Это **финальная архитектура KG**, используемая
    в гибридной модели.
    """)

    col1, col2 = st.columns([3, 1])
    with col1:
        displayed = display_static_kg_viz()
        if not displayed:
            st.info("Используется та же визуализация, что и для KG V1 (показана выше)")

    with col2:
        st.markdown("**14 Thread-Level Graph Features**")
        st.markdown("""
        **Топология**:
        - thread_depth, num_nodes, num_edges
        - avg_branching_factor, max_branching_factor
        - source_reply_count, leaf_ratio
        - avg_depth

        **PageRank**:
        - source_pagerank, avg_pagerank

        **Центральность**:
        - source_centrality, avg_centrality

        **Пользователи**:
        - user_rumor_ratio, unique_users
        """)

# ============================================================
# TAB 5: KG Statistics
# ============================================================
with tabs[4]:
    st.subheader("Детальная статистика Графа знаний")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Процесс построения")
        st.markdown("""
        **1. Определение онтологии** (файл TTL)
        - Классы и свойства определены в OWL/RDF
        - Версия 1: 5 классов, 5 Object Properties
        - Версия 2: 7 классов, расширенные Data Properties

        **2. Генерация RDF Triple** (`build_kg.py`)
        - Класс `KnowledgeGraphBuilder` обрабатывает CSV признаков
        - Создание URI для сообщений, пользователей, событий, threads
        - Валидация: без дубликатов, правильная структура, ацикличность

        **3. Обнаружение и удаление циклов**
        - Обнаружение циклов на основе DFS в графе ответов
        - Обнаружение Self-loop
        - Перекрёстная валидация Thread

        **4. Извлечение признаков графа** (`graph_features.py`)
        - Граф ответов → NetworkX DiGraph
        - Вычисление PageRank, Centrality, Betweenness
        - Анализ репутации пользователей и структуры Thread
        """)

    with col2:
        st.markdown("### Результаты валидации")
        st.markdown("""
        **После исправления URI (05A.1)**:

        | Проверка | Результат |
        |---------|---------|
        | Узлы графа ответов | 76,066 |
        | Рёбра графа ответов | 65,565 |
        | Колонки графа (14) | Все присутствуют |
        | Ненулевая топология | Подтверждено |
        | Обнаружено циклов | 0 (полностью ацикличен) |
        | Обогащённый набор | 102,440 x 31 |

        **Корневая причина (05A)**:
        - Парсинг URI проверял `'/post/'`, но реальные URI использовали `'#post/'`
        - Исправлено в `utils/graph_features.py` строка 34
        - До исправления: 0 рёбер. После исправления: 65,565 рёбер.
        """)

st.markdown("---")
st.caption("Граф знаний построен из набора данных PHEME с использованием RDFLib и NetworkX")