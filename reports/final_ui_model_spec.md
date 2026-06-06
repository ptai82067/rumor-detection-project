# Final UI Model Specification

> **Date**: 2026-06-07  
> **Purpose**: Exact feature composition, inference pipeline, UI requirements

---

## 1. V1 Model — Post-Level (System D from 05B)

### Feature Composition

| Component | Dimension | Source Code Reference | Included in Final Model? |
|-----------|-----------|----------------------|---------------------------|
| TF-IDF | **5,000** | `scripts/train_and_save_v1.py` line 128: `max_features=5000` | ✅ Yes |
| MiniLM | **384** | `scripts/train_and_save_v1.py` line 142: `all-MiniLM-L6-v2` | ✅ Yes |
| Graph | **14** | `scripts/train_and_save_v1.py` line 81: `get_graph_feature_columns()` | ✅ Yes |
| Propagation | **4** | `scripts/train_and_save_v1.py` line 167: `is_reply, thread_size, children_count, depth` | ❌ **NOT USED in V1 System D** |
| **Total** | **5,398** | `X_train_D = np.hstack([X_train_tfidf.toarray(), X_bert_train, X_train_graph])` (line 261) | — |

### CRITICAL FINDING: V1 does NOT use Propagation features

Lines 261-262 in `scripts/train_and_save_v1.py`:
```python
X_train_D = np.hstack([X_train_tfidf.toarray(), X_bert_train, X_train_graph])
#                      ^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^ ^^^^^^^^^^^^^^
#                         5000 TF-IDF          384 MiniLM     14 Graph
#                     NO PROPAGATION FEATURES HERE
```

Propagation features (4 columns: `is_reply`, `thread_size`, `children_count`, `depth`) are computed in Step 3d but used ONLY in System A (baseline). They are **excluded from the final deployment model (System D)**.

This matches the original 05B notebook (`notebooks/05B_bert_graph_fusion_fixed.py` line 247-248).

### V1 Inference Pipeline (for UI)

```
User Input: single text string
    │
    ├── Step 1: TF-IDF
    │   tfidf_vectorizer.transform([text]) → 5,000-dim dense array
    │
    ├── Step 2: MiniLM Embedding
    │   SentenceTransformer('all-MiniLM-L6-v2').encode([text]) → 384-dim
    │
    ├── Step 3: Graph Features
    │   Need: post_id lookup into pheme_features_with_graph.csv
    │   → 14-dim → scaler_graph.transform()
    │   ⚠️ For new/unseen posts: requires computing graph features on-the-fly
    │     (see utils/graph_features.py: extract_all_graph_features)
    │
    └── Step 4: Concatenate + Predict
        np.hstack([tfidf_dense(5000), minilm(384), graph(14)]) → 5,398-dim
        model.predict() → "Rumor" / "Non-Rumor"
        model.predict_proba() → confidence score
```

---

## 2. V2 Model — Thread-Level (Full Hybrid from run_ablation.py)

### Feature Composition

| Component | Dimension | Source Code Reference | Included in Final Model? |
|-----------|-----------|----------------------|---------------------------|
| MiniLM | **384** | `scripts/train_and_save_v2.py` line 113: `all-MiniLM-L6-v2` (source text only) | ✅ Yes |
| Propagation | **4** | `scripts/train_and_save_v2.py` line 86: `thread_size, max_depth, avg_depth_prop, reply_rate` | ✅ Yes |
| Graph | **14** | `scripts/train_and_save_v2.py` lines 142-147: 14 thread-level graph features | ✅ Yes |
| **Total** | **402** | `X_full = np.hstack([embeddings, X_prop, X_graph])` (line 157) | — |

### V2 Inference Pipeline (for UI)

```
User selects / provides: thread_id (must exist in graph_features_v2.csv)
    │
    ├── Step 1: MiniLM Embedding
    │   SentenceTransformer('all-MiniLM-L6-v2').encode([source_text]) → 384-dim
    │   source_text = first post in thread (depth=0)
    │
    ├── Step 2: Propagation Features (4)
    │   thread_size     = total posts in thread
    │   max_depth       = max reply depth
    │   avg_depth_prop  = mean depth of all posts
    │   reply_rate      = total_posts / (thread_size + 1)
    │
    ├── Step 3: Graph Features (14)
    │   Lookup by thread_id from graph_features_v2.csv:
    │   thread_depth, num_nodes, num_edges, avg_branching_factor,
    │   max_branching_factor, source_reply_count, leaf_ratio, avg_depth,
    │   source_pagerank, avg_pagerank, source_centrality, avg_centrality,
    │   user_rumor_ratio, unique_users
    │
    └── Step 4: Concatenate + Scale + Predict
        np.hstack([minilm(384), prop(4), graph(14)]) → 402-dim
        scaler.transform() → scaled 402-dim
        model.predict() → "Rumor" / "Non-Rumor"
        model.predict_proba() → confidence score
```

---

## 3. Feature Comparison: V1 vs V2

| Aspect | V1 (Post-Level) | V2 (Thread-Level) |
|--------|-----------------|-------------------|
| **Total Dim** | 5,398 | 402 |
| **TF-IDF** | 5,000 | None |
| **MiniLM** | 384 (full post text) | 384 (source text only) |
| **Graph** | 14 (post-level from `pheme_features_with_graph.csv`) | 14 (thread-level from `graph_features_v2.csv`) |
| **Propagation** | 0 (❌ NOT USED in System D) | 4 |
| **Classifier** | LogisticRegression(class_weight='balanced') | LogisticRegression(no class_weight) |
| **Accuracy** | 96.23% | 98.11% |
| **Recall** | 96.06% | 96.46% |
| **Input required** | Post text + Post ID (for graph features) | Thread ID (or source text + thread data) |
| **Best for** | Single post prediction | Thread/headline prediction |

---

## 4. UI Input Requirements

### V1: Single Post Prediction

| Input | Required? | Source | How Obtained |
|-------|-----------|--------|-------------|
| `text` | ✅ **Required** | User types/pastes text | Text input widget |
| `post_id` | ❌ Optional | Dataset or auto-generated | If from dataset: dropdown; if new: must compute graph features on-the-fly |
| Graph features | ✅ **Required** | `pheme_features_with_graph.csv` or computed via `graph_features.py` | Lookup by `post_id` for known posts; compute via `extract_all_graph_features()` for new posts |

### V2: Thread Prediction

| Input | Required? | Source | How Obtained |
|-------|-----------|--------|-------------|
| `thread_id` | ✅ **Required** | Pre-loaded from dataset | Dropdown/list selection in UI |
| `source_text` | ✅ **Required** | First post in thread | Loaded from dataset after thread_id selected |
| Thread-level data | ✅ **Required** | Dataset + `graph_features_v2.csv` | Pre-computed, loaded by thread_id |
| Propagation features | ✅ **Required** | Computed from post-level data | Pre-computed, loaded by thread_id |

---

## 5. Recommended UI Controls

| Control | Purpose | Affects |
|---------|---------|---------|
| **Model selector** (radio: V1 / V2) | Choose which classifier to use | All subsequent behavior |
| **Input mode** (radio: Sample / Custom) | Choose input source | What fields to show |
| **Thread dropdown** (selectbox) | Choose sample thread from dataset (V2 mode) | Thread ID for feature lookup |
| **Post dropdown** (selectbox) | Choose sample post from dataset (V1 mode) | Post ID for feature lookup |
| **Text area** (text_input) | Type/paste tweet text (V1 custom mode) | TF-IDF + MiniLM features |
| **Run Detection button** | Trigger prediction | All |
| **Results panel** (read-only) | Display prediction + confidence | — |
| **Feature explorer** (expander) | Show feature values for current input | — |

---

## 6. Recommended Demo Workflow

### Default Mode: V2 Thread Prediction

1. User selects model: **V2** (pre-selected, best accuracy)
2. User picks a thread from dropdown (e.g., `thread_id=498254929942028288`)
3. System loads pre-computed features from:
   - `graph_features_v2.csv` (14 graph features)
   - `pheme_features.csv` (propagation features: thread_size, max_depth, etc.)
   - Source text from depth=0 post
4. MiniLM generates embedding from source text
5. Prediction runs (402-dim → scaler → model → result)
6. UI shows: [Rumor/Non-Rumor], confidence %, feature values

### Alternative: V1 Custom Text

1. User switches to **V1** model
2. User types/pastes tweet text
3. User selects a post_id from dataset (or enters one manually for graph feature lookup)
4. TF-IDF vectorizer transforms text → 5,000-dim
5. MiniLM generates embedding → 384-dim
6. Graph features looked up by post_id → 14-dim
7. Prediction runs (5,398-dim → model → result)
8. UI shows: [Rumor/Non-Rumor], confidence %, feature values

### Recommended Default

**V2 mode with sample thread selection** provides the best demo experience:
- Fastest (no TF-IDF, no full post-level graph computation)
- Most accurate (98.11%)
- Most visually appealing (show thread reply tree + features)
- Best for thesis defense presentation

---

## 7. Key Implementation Notes

1. **V1 does not use propagation** (4 features). The saved model expects exactly 5,398 features: 5,000 TF-IDF + 384 MiniLM + 14 Graph.
2. **V2 does not use TF-IDF**. The saved model expects exactly 402 features: 384 MiniLM + 4 Propagation + 14 Graph.
3. **Graph features for unknown posts**: If the UI allows entering new text not in the dataset, graph features must be computed via `utils/graph_features.py:extract_all_graph_features(df)` which requires the full PHEME dataframe. This is computationally intensive (~10 seconds). For a demo, restrict graph features to threads/posts already in the dataset.
4. **MiniLM model download**: First run downloads `all-MiniLM-L6-v2` (~80 MB) from HuggingFace. Cache with `@st.cache_resource`.
5. **No model files exist yet** — V1/V2 training scripts must be run to generate artifacts in `models/v1/` and `models/v2/`.