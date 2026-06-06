# UI Planning Report — Streamlit Demo for Rumor Detection Thesis

> **Date**: 2026-06-07  
> **Status**: Pre-implementation audit  
> **Goal**: Generate `docs/ui_planning_report.md` before writing any UI code

---

## Table of Contents

1. [Existing Reusable Components](#1-existing-reusable-components)
2. [Existing Models and Weights](#2-existing-models-and-weights)
3. [Existing Feature Pipelines](#3-existing-feature-pipelines)
4. [Existing Visualization Assets](#4-existing-visualization-assets)
5. [Best Architecture for UI Integration](#5-best-architecture-for-ui-integration)
6. [Recommended Streamlit Page Structure](#6-recommended-streamlit-page-structure)
7. [Data Flow Architecture](#7-data-flow-architecture)
8. [Risks and Dependencies](#8-risks-and-dependencies)
9. [Implementation Steps](#9-implementation-steps)

---

## 1. Existing Reusable Components

### 1.1 Python Modules (Directly Importable)

| Module | Path | Purpose | Reusable For UI |
|--------|------|---------|-----------------|
| `bert_feature_fusion` | `utils/bert_feature_fusion.py` | BERT embedding extraction (384-dim MiniLM), normalization, fusion with graph features | **Yes** — `load_bert_model()`, `extract_sentence_embeddings()`, `get_feature_dimension_breakdown()` |
| `graph_features` | `utils/graph_features.py` | 14 graph features: centrality, user reputation, thread structure, source authority | **Yes** — `extract_all_graph_features()`, `get_graph_feature_columns()`, `get_feature_sets()` |
| `build_kg` | `knowledge_graph/build_kg.py` | KG v1 construction from features, cycle detection, RDF graph building | **Partial** — `KnowledgeGraphBuilder` class for KG statistics display |
| `PropagationFeatureExtractor` | `preprocessing/propagation_features.py` | Thread structure, depth, reply features | **Partial** — for demo input processing |
| `loader` | `preprocessing/loader.py` | PHEME dataset traversal | **Path-only** — not needed for demo |
| `parser` | `preprocessing/parser.py` | Thread parsing from raw data | **Not needed** — demo uses processed data |

### 1.2 Key Function Signatures for UI Integration

```python
# From utils.bert_feature_fusion.py:
load_bert_model(model_name="all-MiniLM-L6-v2") -> SentenceTransformer
extract_sentence_embeddings(texts: List[str], model, batch_size=32) -> np.ndarray (n, 384)
create_fusion_feature_matrix(df_train, df_test, df_full, ...) -> Dict
get_feature_dimension_breakdown() -> Dict[str, int]

# From utils.graph_features.py:
extract_all_graph_features(df, kg_path) -> pd.DataFrame  # +14 columns
get_graph_feature_columns() -> List[str]  # 14 column names
get_feature_sets() -> Dict[str, List[str]]  # Feature subsets for ablation

# From src.extract_graph_features_v2.py:
compute_all_features(df) -> pd.DataFrame  # 14 thread-level features (5802 rows)
```

### 1.3 Pre-computed Data Artifacts

| File | Path | Size | Content |
|------|------|------|---------|
| Base features | `data/processed/pheme_features.csv` | ~15 MB | 102,440 posts, ~20 columns (text, label, propagation) |
| Features + graph | `data/processed/pheme_features_with_graph.csv` | ~18 MB | Same + 14 graph feature columns |
| Graph features v2 | `data/processed/graph_features_v2.csv` | ~200 KB | 5,802 threads, 15 columns (thread-level) |
| MiniLM embeddings | `data/processed/minilm_embeddings_thread.npy` | ~9 MB | Pre-computed 384-dim embeddings |
| KG v1 | `data/processed/pheme_kg.ttl` | ~3 MB | RDF triples for reply graph |
| KG v2 | `data/processed/pheme_kg_v2.ttl` | ~5 MB | Extended RDF triples |
| Clean data | `data/processed/pheme_clean.csv` | ~15 MB | Preprocessed raw tweets |

---

## 2. Existing Models and Weights

### 2.1 Trained Model Status

> **⚠️ No pre-saved model files exist in `models/` directory.**  
> The `models/` folder at project root is **empty**.

All models are trained **on-the-fly** during notebook execution using `scikit-learn`'s `LogisticRegression`. No weights have been serialized (no `.joblib`, `.pkl`, or `.pth` files).

### 2.2 Model Configurations (from ablation study)

| Config | Features | Acc | Recall | F1 | Pipeline Source |
|--------|----------|-----|--------|-----|-----------------|
| TF-IDF | 10K TF-IDF | 0.8269 | 0.6101 | 0.7057 | `first_baseline.py` |
| MiniLM only | 384 BERT | 0.8338 | 0.7468 | 0.7535 | `run_bert_graph_fusion.py` |
| + Propagation | 384+4 | 0.8355 | 0.7468 | 0.7554 | `run_bert_graph_fusion.py` |
| + Graph (KG v2) | 384+14 | **0.9811** | **0.9646** | **0.9719** | `run_bert_graph_fusion.py` |
| Full Hybrid | 384+4+14 | **0.9811** | **0.9646** | **0.9719** | `run_bert_graph_fusion.py` |

### 2.3 Model Serialization Strategy for UI

The UI must either:
- **Option A**: Train a model on startup (loads data, trains LogisticRegression) — takes seconds
- **Option B**: Pre-train and save models before UI launch — preferable for demo responsiveness
- **Option B recommended**: Add a `save_model()` step in the pipeline, then load in the UI

Required artifacts for offline prediction mode:
```
models/
  full_hybrid_model.joblib          # Trained LogisticRegression (optional)
  tfidf_vectorizer.joblib            # For text vectorization
  scaler_propagation.joblib          # Propagation feature scaler
  scaler_graph.joblib                # Graph feature scaler
```

---

## 3. Existing Feature Pipelines

### 3.1 Complete Processing Pipeline

```
Raw PHEME data
    ↓ preprocessing/loader.py + parser.py + normalizer.py
pheme_clean.csv  (102,440 posts, cleaned)
    ↓ preprocessing/propagation_features.py
pheme_features.csv  (+ depth, children_count, thread_size, reply_speed, ...)
    ↓ utils/graph_features.py (loads KG -> networkx -> compute)
pheme_features_with_graph.csv  (+ 14 graph features: pagerank, centrality, etc.)
    ↓ src/extract_graph_features_v2.py  (thread-level aggregation)
graph_features_v2.csv  (14 thread-level features, 5802 rows)
    ↓ utils/bert_feature_fusion.py  (384-dim MiniLM embeddings)
minilm_embeddings_thread.npy
```

### 3.2 Feature Groups for UI Display

| Group | Columns | Count | Type |
|-------|---------|-------|------|
| **Propagation** | `is_reply`, `thread_size`, `children_count`, `depth` | 4 | Post-level |
| **Node Centrality** | `node_in_degree`, `node_out_degree`, `pagerank_score` | 3 (+2 extended) | Post-level |
| **User Reputation** | `user_prior_rumor_ratio`, `user_post_count`, `user_thread_count`, `user_avg_depth` | 4 | Post-level |
| **Thread Structure** | `subtree_reply_count`, `sibling_count`, `position_in_thread` | 3 | Post-level |
| **Source Authority** | `source_user_credibility`, `source_network_size` | 2 | Post-level |
| **Thread-Level v2** | `thread_depth`, `num_nodes`, `num_edges`, `avg_branching_factor`, `max_branching_factor`, `source_reply_count`, `leaf_ratio`, `avg_depth`, `source_pagerank`, `avg_pagerank`, `source_centrality`, `avg_centrality`, `user_rumor_ratio`, `unique_users` | 14 | Thread-level |

---

## 4. Existing Visualization Assets

### 4.1 Pre-generated Figures

| Asset | Path | Format | Size | Description |
|-------|------|--------|------|-------------|
| KG Semantic Final | `visualization/kg_semantic_final.png` | PNG | 414 KB | Full ontology + KG visualization with 7 relation types |
| KG Semantic Final | `visualization/kg_semantic_final.svg` | SVG | 173 KB | Vector version of above |
| Ablation Bar Chart | `results/ablation/figures/fig1_ablation_bar_metrics.png` | PNG | — | Bar chart comparing accuracy, recall, F1 |
| Recall/FN Trend | `results/ablation/figures/fig2_recall_fn_trend.png` | PNG | — | Recall vs FN improvement across configs |
| Confusion Matrices | `results/ablation/figures/fig3_confusion_matrices.png` | PNG | — | Multi-panel confusion matrix comparison |
| Feature Contribution | `results/ablation/figures/fig4_feature_contribution.png` | PNG | — | Feature importance contribution analysis |

### 4.2 Generated Diagram Files

| File | Path | Description |
|------|------|-------------|
| Mermaid Class Diagram | `ontology_mermaid_class_diagram.md` | Class diagram in Mermaid syntax |
| Mermaid ER Diagram | `ontology_mermaid_er_diagram.md` | ER diagram in Mermaid syntax |

### 4.3 Visualization Scripts

| Script | Path | Purpose |
|--------|------|---------|
| `viz_kg_semantic_final.py` | `notebooks/viz_kg_semantic_final.py` | Generates KG visualization with matplotlib+networkx |
| `generate_ontology_diagram.py` | Root | Generates ontology diagrams |
| `generate_mermaid_diagram.py` | Root | Generates Mermaid syntax diagrams |

---

## 5. Best Architecture for UI Integration

### 5.1 Recommended Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Framework** | **Streamlit** | Simpler than Gradio for multi-page academic demos; better layout control |
| **Data loading** | `pandas` + `numpy` | Already used throughout project |
| **ML inference** | `scikit-learn` + `sentence-transformers` | Already in requirements.txt |
| **Graph visualization** | `streamlit-agraph` OR `pyvis` | Interactive graph rendering in browser |
| **Charts** | `plotly` (already in requirements) | Interactive plots for metrics display |
| **Caching** | `@st.cache_data` | Streamlit built-in caching for data and model loading |

### 5.2 Directory Structure

```
ui/
  __init__.py
  app.py                          # Main entry point
  pages/
    __init__.py
    page1_overview.py             # Project Overview
    page2_demo.py                 # Rumor Detection Demo
    page3_graph_features.py       # Graph Features Display
    page4_kg_viz.py               # KG Visualization
    page5_experiment_results.py   # Experiment Results
  components/
    __init__.py
    data_loader.py                # Shared data loading with caching
    model_manager.py              # Model loading/prediction
    kg_visualizer.py              # Interactive KG visualization
    metrics_charts.py             # Reusable chart components
  assets/
    logo.png                      # Optional project logo
    screenshots/                  # For thesis appendix
```

### 5.3 Key Architectural Decisions

1. **Lazy Loading**: Load pre-computed data once via `@st.cache_data` — no heavy recomputation
2. **Offline Mode**: Train models at startup or load pre-saved `.joblib` files
3. **Demo Mode**: For the detection demo, sample a subset of threads and cache predictions
4. **Interactive KG**: Use `pyvis` (HTML output) for interactive graph; fallback to static matplotlib PNG
5. **Separate Concerns**: `components/` holds reusable widgets; `pages/` holds page layout

---

## 6. Recommended Streamlit Page Structure

### Page 1: Project Overview (page1_overview.py)

**Purpose**: Introduce the project, ontology, and dataset at a glance.

**Content**:
- Project title and abstract (2-3 sentences about rumor detection + KG)
- **Ontology card**: Quick summary of ontology v1/v2 classes and relations
- **Dataset statistics**: Key numbers loaded from `results/pheme_dataset_statistics.csv`
  - 5 events, 5,802 threads, 103,212 tweets, 49,345 users
  - 33.99% rumor / 66.01% non-rumor distribution
- **Pipeline flowchart**: Simple diagram showing data → features → KG → fusion → prediction
- **Key metrics banner**: Baseline recall (61%) → Full hybrid recall (96.46%)

**Assets Used**:
- `results/pheme_dataset_statistics.csv`
- `ontology/pheme_ontology_v1.ttl` (parsed for class display)
- `final_metrics_table.csv`
- Pre-generated Mermaid diagrams

### Page 2: Rumor Detection Demo (page2_demo.py)

**Purpose**: Interactive text/thread input → prediction with confidence score.

**Content**:
- **Input mode selector**:
  - *Single text*: Text area for typing/pasting a tweet
  - *Thread example*: Dropdown to select a sample thread from dataset
- **Feature display**: Show extracted features for the input (optional expander)
- **Run Detection button**: Executes prediction pipeline
- **Results panel**:
  - Prediction label: "Rumor" (red) or "Non-Rumor" (green)
  - Confidence score with progress bar (0-100%)
  - Probability breakdown (pie or bar chart)
- **Model selector** (optional sidebar): Choose which model configuration to use

**Pipeline for demo**:
1. Load pre-computed features + graph data
2. On input: generate TF-IDF vector, extract BERT embedding, look up graph features
3. Concatenate features → predict with pre-trained LogisticRegression
4. Display results

**Assets Used**:
- `data/processed/pheme_features_with_graph.csv`
- `data/processed/graph_features_v2.csv`
- Pre-trained model (trained at startup or loaded from disk)
- `utils/bert_feature_fusion.py`
- `utils/graph_features.py`

### Page 3: Graph Features (page3_graph_features.py)

**Purpose**: Display extracted graph features and topology statistics.

**Content**:
- **Feature group selector**: Dropdown or tabs for each group (Node Centrality, User Reputation, Thread Structure, Source Authority)
- **Summary statistics table**: For each group, show min, max, mean, std across dataset
- **Distribution histogram**: Plotly histogram for selected feature
- **Thread-level features section**: Display `graph_features_v2.csv` content
  - Sortable table with 14 feature columns
  - Statistics summary
- **Correlation heatmap**: Plotly heatmap showing correlation between graph features and label

**Assets Used**:
- `utils/graph_features.py` (get_feature_sets(), get_graph_feature_columns())
- `data/processed/graph_features_v2.csv`
- `data/processed/pheme_features_with_graph.csv`

### Page 4: Knowledge Graph Visualization (page4_kg_viz.py)

**Purpose**: Interactive display of the knowledge graph structure.

**Content**:
- **Static visualization**: Display `visualization/kg_semantic_final.png` with explanatory caption
- **Interactive graph**: Using `pyvis` or `streamlit-agraph` to render a sampled subgraph
  - Allow selecting a thread ID to visualize its reply tree
  - Color nodes by: label (rumor/non-rumor), depth, user
- **Ontology viewer**: Display ontology classes and properties from TTL files
- **Graph statistics**: Total triples, nodes, edges from KG

**Assets Used**:
- `visualization/kg_semantic_final.png`
- `visualization/kg_semantic_final.svg`
- `data/processed/pheme_kg.ttl`
- `knowledge_graph/build_kg.py` (metadata extraction)

### Page 5: Experiment Results (page5_experiment_results.py)

**Purpose**: Final metrics, ablation study, comparison charts.

**Content**:
- **Final metrics table**: Display `final_metrics_table.csv` as a formatted table
  - 5 models: TF-IDF+Prop, TF-IDF+Graph, MiniLM only, MiniLM+Graph, Full Hybrid
  - 7 metrics: Accuracy, Precision, Recall, F1, ROC-AUC, MCC, FN
- **Ablation table**: Display `results/ablation/ablation_table.csv`
- **Pre-generated figures**: Show all 4 ablation figures from `results/ablation/figures/`
- **Interactive charts** (optional, can supplement static figures):
  - Plotly bar chart comparing model metrics
  - Plotly line chart showing recall progression
  - Confusion matrix heatmaps
- **Key insight card**: Highlight that Full Hybrid achieves 96.46% recall (down from 61% baseline)

**Assets Used**:
- `final_metrics_table.csv`
- `results/ablation/ablation_table.csv`
- `results/ablation/figures/*.png`
- `plotly` for interactive overlays

---

## 7. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT APP                            │
│                                                                 │
│  ┌───────────────┐     ┌──────────────────────────────┐        │
│  │  @st.cache_data│     │  components/data_loader.py   │        │
│  │  on startup   │────▶│  - load_features_df()        │        │
│  │               │     │  - load_graph_features_df()  │        │
│  │               │     │  - load_kg_graph()           │        │
│  │               │     │  - load_metrics_table()      │        │
│  └───────────────┘     └───────────┬──────────────────┘        │
│                                    │                           │
│  ┌─────────────────────────────────▼──────────────────────┐    │
│  │              SHARED STATE (st.session_state)           │    │
│  │  df_pheme, df_graph, kg_graph, model, scalers, etc.   │    │
│  └──────┬──────────┬──────────┬──────────┬───────────────┘    │
│         │          │          │          │                     │
│  ┌──────▼──┐ ┌─────▼────┐ ┌──▼────┐ ┌──▼─────────────┐       │
│  │ Page 1  │ │ Page 2   │ │Page 3 │ │  Page 4        │       │
│  │Overview │ │ Demo     │ │Graph  │ │  KG Viz        │       │
│  └─────────┘ └─────┬────┘ └──┬────┘ └──┬─────────────┘       │
│                    │         │         │                      │
│                    │  ┌──────▼──────┐  │                      │
│                    │  │ Model       │  │                      │
│                    │  │ Manager     │  │                      │
│                    │  │ (predict)   │  │                      │
│                    │  └─────────────┘  │                      │
│                    │                   │                      │
│              ┌─────▼───────────────────▼──────┐               │
│              │  components/kg_visualizer.py    │               │
│              │  components/metrics_charts.py   │               │
│              └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Risks and Dependencies

### 8.1 Critical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **No pre-saved models exist** | High | Train on UI startup (takes ~30s); or add pre-training script |
| **BERT model download on first run** | Medium | Cache model; use `@st.cache_resource`; warn user about download |
| **Large data files (~20MB total)** | Low | Use `@st.cache_data`; only load needed columns |
| **KG visualization performance** | Medium | Sample subgraphs; limit nodes to ~50; use static image as fallback |
| **Streamlit not in requirements.txt** | Low | Add to requirements; install via pip |
| **`pyvis`/`streamlit-agraph` compatibility** | Medium | Test first; fallback to `matplotlib` static geaphs |

### 8.2 Dependencies to Add

```
# requirements.txt additions for UI:
streamlit>=1.28.0
pyvis>=0.3.1              # For interactive KG graphs (optional)
plotly>=5.0.0             # Already listed; verify installed
joblib>=1.0.0             # Already listed; for model serialization
```

### 8.3 Compatibility Concerns

- **Python version**: Scripts use Python 3.8+ features; Streamlit requires 3.8+
- **sentence-transformers**: Works on CPU but slower (~2x) than GPU
- **Windows compatibility**: All paths use `os.path.join`; should work on Windows
- **Network requirement**: First run needs internet for BERT model download (~80MB)

---

## 9. Implementation Steps

### Phase 1: Setup (5 min)

1. Add `streamlit` to `requirements.txt`
2. Create `ui/` directory structure
3. Train and save model artifacts (or add startup training)

### Phase 2: Core Infrastructure (15 min)

4. Implement `components/data_loader.py` with `@st.cache_data`
5. Implement `components/model_manager.py` with model loading/prediction
6. Implement `app.py` with multi-page navigation sidebar

### Phase 3: Pages (30 min each)

7. `page1_overview.py` — Project intro, ontology, dataset stats
8. `page2_demo.py` — Input text, predict, confidence display
9. `page3_graph_features.py` — Feature tables, distributions, correlations
10. `page4_kg_viz.py` — Static + interactive KG visualization
11. `page5_experiment_results.py` — Metrics tables, ablation figures, charts

### Phase 4: Polish (10 min)

12. Add academic-styling (clean colors, proper fonts, thesis title)
13. Add explanatory tooltips and captions for thesis context
14. Test all pages; capture screenshots for thesis appendix

**Total estimated time**: ~2-3 hours

---

## Summary

The existing repository contains **all necessary data, feature pipelines, and experiment results** to build a Streamlit UI. The key gaps are:

1. **No serialized model weights** — must train at startup or pre-save
2. **No Streamlit dependency** — must be added
3. **No interactive graph library** — recommend `pyvis`

The proposed 5-page architecture covers:
- **Page 1**: Project overview (ontology, dataset, pipeline)
- **Page 2**: Live detection demo (input → prediction)
- **Page 3**: Graph feature exploration (statistics, distributions)
- **Page 4**: KG visualization (interactive graph)
- **Page 5**: Experiment results (metrics, ablation, charts)

All components are designed to **leave the existing research pipeline completely unchanged**.