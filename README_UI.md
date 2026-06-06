# PHEME Rumor Detection — Thesis Demo UI

## Overview

This Streamlit application provides an interactive demonstration of the graduation thesis:

> **Rumor Detection on Social Media using Ontology and Knowledge Graph**

The demo showcases the complete research pipeline, from ontology design and knowledge graph construction to hybrid feature fusion and classification.

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Install dependencies (from project root)
pip install -r requirements.txt

# Train and save the model artifacts
python scripts/train_and_save_v1.py  # V1: Post-Level model (5,398 dim)
python scripts/train_and_save_v2.py  # V2: Thread-Level model (402 dim)

# Launch the UI
streamlit run ui/app.py
```

### What Each Pipeline Does

| Script | Model | Features | Dimensions | Expected Accuracy |
|--------|-------|----------|-----------|-------------------|
| `train_and_save_v1.py` | V1 (05B) | TF-IDF + MiniLM + Graph | 5,398 | 96.23% |
| `train_and_save_v2.py` | V2 (Ablation) | MiniLM + Propagation + Graph | 402 | 98.11% |

## UI Pages

| Page | Description |
|------|-------------|
| **Home** | Landing page with project overview and quick stats |
| **🔬 Research Evolution** | Complete 3-phase research timeline (Baseline → Graph → Hybrid) |
| **🕵️ Rumor Detection** | Interactive demo — select threads, run predictions, explore features |
| **🕸️ Ontology & KG Explorer** | Ontology V1/V2 classes, KG visualizations, statistics |
| **📊 Feature Analysis** | Detailed feature dimensions, group breakdowns, comparison charts |
| **📈 Experimental Results** | Final metrics, ablation study, confusion matrices, figures |

## Page 2: Rumor Detection Demo

### How to Use

1. **Select Model Version** (sidebar):
   - **V2 (Thread-Level)** — Recommended (402 dim, 98.11% accuracy)
   - **V1 (Post-Level)** — Alternative (5,398 dim, 96.23% accuracy)

2. **Select a Thread** from the dropdown

3. Click **"Analyze Thread"**

4. View:
   - Prediction result (Rumor / Non-Rumor)
   - Confidence score with progress bar
   - Feature count and inference time
   - Full conversation thread display
   - Feature explorer (graph + propagation features)

### V2 Feature Composition

| Component | Dim | Description |
|-----------|-----|-------------|
| MiniLM | 384 | SentenceTransformer embedding of source text |
| Propagation | 4 | thread_size, max_depth, avg_depth, reply_rate |
| Graph | 14 | Topology, PageRank, centrality, user features |
| **Total** | **402** | Concatenated and scaled |

### V1 Feature Composition

| Component | Dim | Description |
|-----------|-----|-------------|
| TF-IDF | 5,000 | Unigrams + bigrams with IDF weighting |
| MiniLM | 384 | SentenceTransformer embedding of post text |
| Graph | 14 | Post-level centrality, reputation, structure |
| **Total** | **5,398** | Concatenated (no propagation in System D) |

## Model Artifacts

After running the training scripts, the following files are generated:

```
models/
├── v1/
│   ├── model.joblib              LogisticRegression (5,398 input dim)
│   ├── tfidf_vectorizer.joblib   TfidfVectorizer (5,000 vocab)
│   ├── scaler_graph.joblib       StandardScaler (14 dim)
│   ├── scaler_propagation.joblib StandardScaler (4 dim)
│   └── metadata.json             Experiment config + metrics
│
└── v2/
    ├── model.joblib              LogisticRegression (402 input dim)
    ├── scaler.joblib             StandardScaler (402 dim)
    └── metadata.json             Experiment config + metrics
```

## Data Pre-loaded

The following datasets are loaded at startup (not regenerated):

- `data/processed/pheme_features.csv` — Base features (102,440 posts)
- `data/processed/pheme_features_with_graph.csv` — Features with graph columns
- `data/processed/graph_features_v2.csv` — Thread-level graph features (5,802 threads)
- `results/pheme_dataset_statistics.csv` — Dataset summary statistics
- `results/ablation/ablation_table.csv` — Ablation study results
- `final_metrics_table.csv` — Final performance metrics
- `visualization/kg_semantic_final.png` — KG visualization
- `results/ablation/figures/*.png` — Ablation study figures

## File Structure

```
ui/
├── app.py                        # Main entry point
├── pages/
│   ├── 1_Research_Evolution.py   # Research timeline
│   ├── 2_Rumor_Detection.py      # Interactive detection demo
│   ├── 3_Ontology_KG_Explorer.py # Ontology + KG visualization
│   ├── 4_Feature_Analysis.py     # Feature composition details
│   └── 5_Experimental_Results.py # Metrics, ablation, confusion matrices
├── components/
│   ├── data_loader.py            # Cached data loading
│   ├── model_manager.py          # Model loading + prediction
│   ├── kg_visualizer.py          # KG image display
│   └── metrics_charts.py         # Plotly chart components
└── assets/                       # (future use)
```

## Screenshots for Thesis

The UI is designed to be screenshot-friendly:
- Academic color scheme
- Clean layout with proper spacing
- Metric cards with clear labels
- High-contrast readability

## Dependencies

See `requirements.txt`. Key additions for the UI:
- `streamlit>=1.28.0` — UI framework
- `pyvis>=0.3.1` — Interactive graph visualization (optional)