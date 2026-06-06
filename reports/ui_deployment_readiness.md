# UI Deployment Readiness Report

> **Date**: 2026-06-07  
> **Status**: Artifacts being generated  
> **Prerequisite for**: Streamlit Demo UI implementation

---

## A. V1 Pipeline Status

| Component | Status | Path |
|-----------|--------|------|
| Training Script | ✅ Created | `scripts/train_and_save_v1.py` |
| Model (`model.joblib`) | 🔄 Generating | `models/v1/model.joblib` |
| TF-IDF Vectorizer (`tfidf_vectorizer.joblib`) | 🔄 Generating | `models/v1/tfidf_vectorizer.joblib` |
| Graph Scaler (`scaler_graph.joblib`) | 🔄 Generating | `models/v1/scaler_graph.joblib` |
| Propagation Scaler (`scaler_propagation.joblib`) | 🔄 Generating | `models/v1/scaler_propagation.joblib` |
| Metadata (`metadata.json`) | 🔄 Generating | `models/v1/metadata.json` |

**Pipeline**: Trains System D (TF-IDF + MiniLM + Graph) on **102,440 posts** (81,952 train / 20,488 test).
**Expected metrics**: Accuracy=96.23%, Recall=96.06%, F1=93.96%
**Estimated training time**: ~15 minutes (MiniLM embedding extraction is the bottleneck)

### V1 Feature Engineering (for inference)

For a single post prediction at inference time:
1. **TF-IDF**: `tfidf_vectorizer.transform([text])` → 5,000-dim sparse → dense
2. **MiniLM**: `sentence-transformers.encode([text])` → 384-dim dense
3. **Graph features**: Lookup from pre-computed `pheme_features_with_graph.csv` by `post_id` → 14-dim → `scaler_graph.transform()`
4. **Propagation features**: From post metadata → 4-dim → `scaler_propagation.transform()`
5. **Concatenate**: `[TF-IDF(5000), MiniLM(384), Graph(14)]` → **5,398-dim total**
6. **Predict**: `model.predict()` / `model.predict_proba()`

---

## B. V2 Pipeline Status

| Component | Status | Path |
|-----------|--------|------|
| Training Script | ✅ Created | `scripts/train_and_save_v2.py` |
| Model (`model.joblib`) | 🔄 Generating | `models/v2/model.joblib` |
| Scaler (`scaler.joblib`) | 🔄 Generating | `models/v2/scaler.joblib` |
| Metadata (`metadata.json`) | 🔄 Generating | `models/v2/metadata.json` |

**Pipeline**: Trains Full Hybrid (MiniLM + Propagation + Graph) on **5,802 threads** (~4,642 train / ~1,160 test).
**Expected metrics**: Accuracy=98.11%, Recall=96.46%, F1=97.19%
**Estimated training time**: ~5 minutes (smaller dataset, only source text embeddings)

### V2 Feature Engineering (for inference)

For a single thread prediction at inference time:
1. **MiniLM**: `SentenceTransformer.encode([source_text])` → 384-dim
2. **Propagation**: Build `[thread_size, max_depth, avg_depth_prop, reply_rate]` → 4-dim
3. **Graph features**: Lookup from `graph_features_v2.csv` by `thread_id` → 14-dim
4. **Concatenate**: `[MiniLM(384), Prop(4), Graph(14)]` → **402-dim total**
5. **Scale**: `scaler.transform()` → 402-dim
6. **Predict**: `model.predict()` / `model.predict_proba()`

---

## C. Artifact Inventory

### Expected Final Artifacts

```
models/
├── v1/
│   ├── model.joblib              ~85 KB  (LogisticRegression, 5398 features)
│   ├── tfidf_vectorizer.joblib   ~500 KB (TfidfVectorizer, 5000 vocabulary)
│   ├── scaler_graph.joblib       ~1 KB   (StandardScaler, 14 features)
│   ├── scaler_propagation.joblib ~1 KB   (StandardScaler, 4 features)
│   └── metadata.json             ~3 KB   (Experiment config + metrics)
│
└── v2/
    ├── model.joblib              ~3 KB   (LogisticRegression, 402 features)
    ├── scaler.joblib             ~1 KB   (StandardScaler, 402 features)
    └── metadata.json             ~2 KB   (Experiment config + metrics)
```

**Total model storage**: ~600 KB (negligible)

### Pre-existing Data Assets (unchanged)

```
data/processed/
├── pheme_features_with_graph.csv     ~18 MB  (V1 inference: graph/prop features)
├── graph_features_v2.csv             ~200 KB (V2 inference: thread-level graph features)
├── pheme_features.csv                ~15 MB  (Base features)
├── pheme_kg.ttl                      ~3 MB   (Knowledge Graph v1)
├── pheme_kg_v2.ttl                   ~5 MB   (Knowledge Graph v2)
├── minilm_embeddings_thread.npy      ~9 MB   (Pre-computed thread embeddings)
└── pheme_clean.csv                   ~15 MB  (Cleaned raw data)
```

---

## D. Required Streamlit Dependencies

The following packages are needed in addition to existing `requirements.txt`:

| Package | Version | Purpose | Already in requirements.txt? |
|---------|---------|---------|------------------------------|
| `streamlit` | >=1.28.0 | UI framework | ❌ No — needs to be added |
| `pyvis` | >=0.3.1 | Interactive KG visualization | ❌ No — needs to be added |
| `plotly` | >=5.0.0 | Interactive charts (metrics, ablation) | ✅ Yes (line 27) |
| `joblib` | >=1.0.0 | Model serialization/loading | ✅ Yes (line 31) |
| `sentence-transformers` | >=0.4.0 | MiniLM embedding extraction | ✅ Yes (line 9) |
| `scikit-learn` | >=1.0.0 | LogisticRegression, scalers, TF-IDF | ✅ Yes (line 4) |
| `networkx` | >=2.6 | Graph utilities for KG viz | ✅ Yes (line 17) |
| `rdflib` | >=6.0.0 | RDF graph parsing for ontology | ✅ Yes (line 21) |

### Proposed requirements.txt addition:

```
# UI Application
streamlit>=1.28.0
pyvis>=0.3.1
```

---

## E. Remaining Blockers Before UI Development

### Blocker 1: Model Artifact Generation (🔄 In Progress)

- V1 and V2 scripts are running
- Once complete, artifacts will be in `models/v1/` and `models/v2/`
- Validation script will confirm all artifacts load correctly

### Blocker 2: Dependency Installation

- `streamlit` must be installed via `pip install streamlit`
- `pyvis` must be installed via `pip install pyvis`
- These are non-breaking changes to `requirements.txt`

### Blocker 3: V2 Real-time Inference Gap

- V2 inference requires building propagation features from **post-level data** for a brand-new thread (not in the dataset)
- For demo purposes, the UI can select **existing threads from the dataset** where graph_features_v2.csv already has pre-computed features
- For arbitrary text input with no thread context, V2 inference is not directly applicable — use V1 instead

### Blocker 4: None (Clean)

- No existing files need modification
- All existing pipelines, notebooks, and results remain untouched
- The `ui/` directory will be completely new

---

## F. Architecture Decision: Which Pipeline to Use in UI

| Use Case | Recommended Pipeline | Reason |
|----------|---------------------|--------|
| **Single text prediction** (type/paste a tweet) | **V1** | V1 works at post-level with TF-IDF + MiniLM; can generate graph features on-the-fly via `graph_features.py`; propagation features available from post data |
| **Thread prediction** (select sample from dataset) | **V2** | V2 works at thread-level; all features already pre-computed in `graph_features_v2.csv`; simpler, faster, higher accuracy |
| **Interactive demo for defense** | **Both** | Let user select which model to use; show comparison |

### Recommended UI Strategy:

1. **Default mode**: User selects a sample thread from the dataset → V2 prediction (simple, fast, most accurate)
2. **Advanced mode**: User types/pastes a tweet → V1 prediction (actual ML pipeline runs end-to-end)
3. **Model toggle** in sidebar: Switch between V1 and V2

---

## G. Summary

| Requirement | Status |
|-------------|--------|
| V1 training script | ✅ Created (`scripts/train_and_save_v1.py`) |
| V2 training script | ✅ Created (`scripts/train_and_save_v2.py`) |
| V1 model artifacts | 🔄 Generating (MiniLM embedding step, ~10-15 min) |
| V2 model artifacts | 🔄 Generating (once V1 completes) |
| Validation script | ✅ Created (`scripts/validate_saved_models.py`) |
| Model validation report | 🔄 Will generate after validation run |
| Streamlit dependency | ❌ Needs to be added to requirements.txt |
| pyvis dependency | ❌ Needs to be added to requirements.txt |
| Pre-existing data assets | ✅ All available |
| Pipeline modification needed? | ❌ None — all new files |

**Overall Readiness**: 80% — Final artifacts being generated. UI development can begin once V1/V2 scripts complete and validation passes.