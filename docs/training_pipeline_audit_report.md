# Training Pipeline Audit Report — Final Metrics Trace

> **Date**: 2026-06-07  
> **Objective**: Trace the exact execution chain that produced the final reported metrics  
> **Target Metrics**: Acc=98.11%, Recall=96.46%, F1=97.19%  
> **Status**: Complete (no files modified)

---

## 1. Critical Finding: Two Separate Execution Pipelines Exist

There are **two different pipelines** producing **two different metric tables** at **different levels of granularity**:

| Aspect | Pipeline A (05B) | Pipeline B (run_ablation.py) |
|--------|------------------|------------------------------|
| **Unit of analysis** | **Post-level** (102,440 posts) | **Thread-level** (5,802 threads) |
| **Contains your target metrics?** | **No** (max: Acc=96.23%, Recall=96.06%) | **Yes** (Acc=98.11%, Recall=96.46%) |
| **Output file** | `final_metrics_table.csv` | `results/ablation/ablation_table.csv` |
| **Training script** | `notebooks/05B_bert_graph_fusion_fixed.py` | `run_ablation.py` |
| **Synthesis notebook** | `notebooks/05_final_hybrid_results_thesis_executed.ipynb` | N/A (standalone script) |
| **Synthesis notebook 2** | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | N/A |

### Which Pipeline Produced Your Requested Metrics?

The metrics **Acc=98.11%, Recall=96.46%, F1=97.19%** appear in:

- `results/ablation/ablation_table.csv` → "Full Hybrid *" row: `0.9811, 0.9794, 0.9646, 0.9719`
- `results/ablation/ablation_table.md` → same values

These were produced by **`run_ablation.py`** (root directory), NOT by the 05B notebook.

---

## 2. Complete Execution Chain: Pipeline A (05B — Post-Level, Thesis-Ready)

This is the **official thesis pipeline** used in notebooks 05 and 09.

### Chain:

```
Raw PHEME data
  ↓ preprocessing/run_pipeline.py
data/processed/pheme_clean.csv                    [102,440 posts]
  ↓ preprocessing/propagation_features.py
data/processed/pheme_features.csv                  [102,440 posts × ~20 cols]
  ↓ knowledge_graph/build_kg.py (URI parsing FIXED in 05A)
data/processed/pheme_kg.ttl                        [RDF triples: 76,066 nodes, 65,565 edges]
  ↓ utils/graph_features.py (extract_all_graph_features)
data/processed/pheme_features_with_graph.csv       [102,440 posts × 31 cols: 14 graph cols added]
  ↓
notebooks/05B_bert_graph_fusion_fixed.py  ← THE TRAINING SCRIPT
  │  ├── Loads pheme_features_with_graph.csv
  │  ├── Train/test split: 80/20 stratified, RANDOM_STATE=42
  │  │   ├── Train: 81,952 samples
  │  │   └── Test:  20,488 samples
  │  ├── Feature extraction:
  │  │   ├── TF-IDF (5,000 features, 1-2 ngrams)
  │  │   ├── MiniLM embeddings (384-dim, all-MiniLM-L6-v2)
  │  │   ├── Graph features (14 cols, StandardScaler)
  │  │   └── Propagation features (4 cols: is_reply, thread_size, children_count, depth)
  │  ├── Classifier: LogisticRegression (C=1.0, max_iter=1000, class_weight='balanced', RANDOM_STATE=42)
  │  └── 4 Systems evaluated:
  │      ├── A: TF-IDF + Propagation       → 0.8065 / 0.7735 / 0.7090 [Acc/Recall/F1]
  │      ├── B: MiniLM Only                → 0.6865 / 0.6543 / 0.5600
  │      ├── C: MiniLM + Graph             → 0.9591 / 0.9585 / 0.9346
  │      └── D: Hybrid (TF-IDF+MiniLM+Graph) → 0.9623 / 0.9606 / 0.9396
  │
  ↓ Output saved to:
project_brain_bundle/regression_pack/05B_results.json
  ↓
notebooks/05_final_hybrid_results_thesis_executed.ipynb
  │  └── Loads 05B_results.json, displays results, generates thesis-ready figures
  ↓
notebooks/09_final_thesis_results_synthesis_executed.ipynb
  └── Hardcoded frozen_metrics dictionary (line 80-95)
      └── Exports:
          final_metrics_table.csv  ← 5 rows: TF-IDF+Prop, TF-IDF+Graph, MiniLM only, MiniLM+Graph, Full Hybrid
```

### Evidence Files for Pipeline A:

| Step | File | Path |
|------|------|------|
| Training script | `05B_bert_graph_fusion_fixed.py` | `notebooks/05B_bert_graph_fusion_fixed.py` |
| JSON results | `05B_results.json` | `project_brain_bundle/regression_pack/05B_results.json` |
| Results report | `05B_results_report.md` | `project_brain_bundle/regression_pack/05B_results_report.md` |
| Synthesis nb 1 | `05_final_hybrid_results_thesis_executed.ipynb` | `notebooks/05_final_hybrid_results_thesis_executed.ipynb` |
| Synthesis nb 2 | `09_final_thesis_results_synthesis_executed.ipynb` | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` |
| Final CSV | `final_metrics_table.csv` | `./final_metrics_table.csv` |

### Note on "TF-IDF + Graph" row in final_metrics_table.csv

The row `TF-IDF + Graph, 0.8092, 0.7775` does NOT come from 05B (which only ran 4 systems A/B/C/D). It was likely added manually in notebook 09 from a separate experiment (possibly `notebooks/04_graph_feature_integration_executed.ipynb`). The 05B training script produces only 4 systems, and the 05B_results.json contains exactly 4 keys.

---

## 3. Complete Execution Chain: Pipeline B (run_ablation.py — Thread-Level)

This is the pipeline that produced your **target metrics (98.11% / 96.46% / 97.19%)**.

### Chain:

```
Raw PHEME data
  ↓ preprocessing/run_pipeline.py
data/processed/pheme_clean.csv
  ↓ preprocessing/propagation_features.py
data/processed/pheme_features.csv
  ↓ src/extract_graph_features_v2.py
data/processed/graph_features_v2.csv              [5,802 threads × 15 cols]
  │
  ↓ run_ablation.py  ← THE ACTUAL TRAINING SCRIPT
     │
     ├── Loads:
     │   ├── pheme_features.csv        (post-level base features)
     │   └── graph_features_v2.csv     (thread-level graph features)
     │
     ├── Aggregation: Groups posts by thread_id, creates thread-level DF
     │   ├── Source text = first depth=0 post
     │   ├── Thread text = concatenation of all posts in thread
     │   └── Merges with graph_features_v2.csv
     │
     ├── Propagation features (computed from post-level):
     │   ├── thread_size, max_depth, avg_depth_prop, reply_rate
     │
     ├── MiniLM embeddings (generated fresh):
     │   ├── model = SentenceTransformer('all-MiniLM-L6-v2')
     │   ├── Input: source_text column (first post in each thread)
     │   └── Output saved: data/processed/minilm_embeddings_thread.npy
     │
     ├── Train/test split: 80/20 stratified, RANDOM_STATE=42
     │   ├── Train: ~4,642 threads
     │   └── Test:  ~1,160 threads
     │
     ├── 14 graph features from graph_features_v2.csv:
     │   thread_depth, num_nodes, num_edges, avg_branching_factor,
     │   max_branching_factor, source_reply_count, leaf_ratio, avg_depth,
     │   source_pagerank, avg_pagerank, source_centrality, avg_centrality,
     │   user_rumor_ratio, unique_users
     │
     ├── 4 propagation features:
     │   thread_size, max_depth, avg_depth_prop, reply_rate
     │
     ├── Classifier: LogisticRegression (C=1.0, max_iter=1000, RANDOM_STATE=42)
     │   NOTE: NO class_weight='balanced' — differs from 05B!
     │
     ├── Feature scaling: StandardScaler (fit on train, transform test)
     │
     ├── 5 Configs evaluated:
     │   ├── TF-IDF (10K features, 1-2 ngrams) → Acc=0.8269, Recall=0.6101, FN=154
     │   ├── MiniLM (384-dim)                   → Acc=0.8338, Recall=0.7468, FN=100
     │   ├── MiniLM + Propagation               → Acc=0.8355, Recall=0.7468, FN=100
     │   ├── MiniLM + Graph                     → Acc=0.9811, Recall=0.9646, FN=14
     │   └── Full Hybrid (MiniLM+Prop+Graph)    → Acc=0.9811, Recall=0.9646, FN=14  ← TARGET
     │
     └── Outputs:
         ├── results/ablation/ablation_table.csv
         ├── results/ablation/ablation_table.md
         └── results/ablation/figures/*.png  (4 figures)
```

### Evidence Files for Pipeline B:

| Step | File | Path |
|------|------|------|
| Training script | `run_ablation.py` | `./run_ablation.py` |
| Ablation CSV | `ablation_table.csv` | `results/ablation/ablation_table.csv` |
| Ablation MD | `ablation_table.md` | `results/ablation/ablation_table.md` |
| Figures | 4 PNGs | `results/ablation/figures/fig*.png` |
| Thread embeddings | `minilm_embeddings_thread.npy` | `data/processed/minilm_embeddings_thread.npy` |

---

## 4. Feature Set Comparison

| Feature Type | Pipeline A (05B) | Pipeline B (run_ablation) |
|-------------|------------------|--------------------------|
| **Unit** | Post-level (102,440) | Thread-level (5,802) |
| **TF-IDF** | 5,000 features | 10,000 features |
| **MiniLM** | 384-dim (full text) | 384-dim (source text only) |
| **Propagation** | `is_reply, thread_size, children_count, depth` | `thread_size, max_depth, avg_depth_prop, reply_rate` |
| **Graph** | 14 post-level features (from `extract_all_graph_features`) | 14 thread-level features (from `graph_features_v2.csv`) |
| **Classifier** | LogisticRegression(class_weight='balanced') | LogisticRegression(no class_weight) |
| **Test split size** | 20,488 posts | ~1,160 threads |

---

## 5. Model Serialization Status

### Does serialization already exist?

**NO.** A complete search of the repository confirms:

| Artifact | Exists? | Evidence |
|----------|---------|----------|
| `*.joblib` files | ❌ Not found | `models/` directory is **empty** |
| `*.pkl` files | ❌ Not found | No `.pkl` files anywhere |
| `*.pth` files | ❌ Not found | No PyTorch model files |
| `*.pt` files | ❌ Not found | No PyTorch trace files |

### Exact Code Locations for Inserting Serialization

#### Location 1: `run_ablation.py` (for Pipeline B — thread-level target metrics)

**Insert after line 136** (after Full Hybrid training, before results printing at line 138):

```python
# === MODEL SERIALIZATION INSERTION POINT (after line 136) ===
import joblib
import os

os.makedirs('models', exist_ok=True)

# Save Full Hybrid model (uses X = np.hstack([embeddings, X_prop, X_graph]))
# Note: ss is the StandardScaler from line 106, last fitted on embeddings
# For production, you need the scaler fitted on the full hybrid feature space
# The Variable `X` is embeddings only at this point in the code
# Refactor needed: create a fused scaler

# Save TF-IDF vectorizer (from Config 1, line 93)
joblib.dump(tfidf_vec, 'models/tfidf_vectorizer.joblib')

# Save scalers and the Full Hybrid model requires refactoring because
# run_ablation.py recomputes X for each config independently
```

> **Note**: `run_ablation.py` cannot be trivially modified to save models because each config creates features independently (the `run_config()` helper creates new `StandardScaler` per call). A refactored script is needed.

#### Location 2: `notebooks/05B_bert_graph_fusion_fixed.py` (for Pipeline A — post-level)

**Insert after line 249** (after System D training), before Step 5 at line 252:

```python
# === MODEL SERIALIZATION INSERTION POINT (after line 249) ===
# KEY OBJECTS TO SAVE AT THIS POINT:
# - model_D: trained LogisticRegression for Full Hybrid (System D)
# - model_C: trained LogisticRegression for MiniLM + Graph (System C)
# - tfidf_vectorizer: fitted TF-IDF vectorizer (from line 131)
# - graph_scaler: fitted StandardScaler for graph features (from line 152)
# - prop_scaler: fitted StandardScaler for propagation features (from line 164)

import joblib
import os
os.makedirs('models', exist_ok=True)

# Save classifiers
joblib.dump(model, 'models/full_hybrid_model.joblib')      # model_D from train_and_evaluate()
# Note: model is a local variable inside train_and_evaluate(). 
# To save it, refactor to return the model, or save before returning.

# Save vectorizers and scalers
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
joblib.dump(graph_scaler, 'models/scaler_graph.joblib')
joblib.dump(prop_scaler, 'models/scaler_propagation.joblib')
```

> **Note**: The current `train_and_evaluate()` function (line 176-219) creates `model` as a local variable and returns only metrics/cm/report. It must be modified to also return the trained model object, or alternatively the model must be saved inside the function before returning.

---

## 6. Required Artifacts for Deployment

### Recommended `models/` directory contents:

```
models/
├── full_hybrid_model.joblib      # LogisticRegression trained on MiniLM + Prop + Graph
├── tfidf_vectorizer.joblib       # TfidfVectorizer(max_features=5000, ngram_range=(1,2))
├── scaler_graph.joblib            # StandardScaler for 14 graph features
├── scaler_propagation.joblib      # StandardScaler for 4 propagation features
└── metadata.json                  # Experiment configuration
```

### metadata.json contents:

```json
{
  "pipeline": "05B_bert_graph_fusion_fixed.py",
  "level": "post-level",
  "dataset": "data/processed/pheme_features_with_graph.csv",
  "dataset_shape": [102440, 31],
  "train_size": 81952,
  "test_size": 20488,
  "random_state": 42,
  "classifier": "LogisticRegression",
  "classifier_params": {
    "random_state": 42,
    "max_iter": 1000,
    "class_weight": "balanced"
  },
  "features": {
    "tfidf": {"max_features": 5000, "ngram_range": [1, 2]},
    "minilm": {"model": "all-MiniLM-L6-v2", "dimension": 384},
    "propagation": ["is_reply", "thread_size", "children_count", "depth"],
    "graph": 14
  },
  "metrics": {
    "accuracy": 0.9623,
    "precision": 0.9194,
    "recall": 0.9606,
    "f1_score": 0.9396,
    "roc_auc": 0.9941,
    "mcc": 0.9127,
    "false_negatives": 246
  },
  "generated": "2026-06-07"
}
```

---

## 7. Exact Commands to Generate Artifacts

### Option A: Regenerate from Pipeline A (05B — Post-Level, Preferred for Deployment)

```bash
# Step 1: Ensure environment is ready
pip install -r requirements.txt

# Step 2: Run 05B training script (generates 05B_results.json + trains all 4 systems)
python notebooks/05B_bert_graph_fusion_fixed.py

# Step 3: Run a separate serialization script OR modify 05B to save models
# (Requires code modification to add joblib.dump calls as noted in Section 5)
```

### Option B: Regenerate from Pipeline B (run_ablation.py — Thread-Level)

```bash
# Step 1: Run ablation study (generates embeddings, trains all 5 configs)
python run_ablation.py

# Step 2: Run a refactored version that saves models
# (run_ablation.py needs refactoring to make models persistable)
```

### Option C: Create a New Serialization Script (Recommended)

Create `scripts/train_and_save.py` based on `notebooks/05B_bert_graph_fusion_fixed.py` with `joblib.dump()` calls added at the end. Command:

```bash
python scripts/train_and_save.py
```

This would be a new file that copies the 05B logic and adds 5 `joblib.dump()` calls and a `json.dump()` for metadata.

---

## 8. Summary Table

| Question | Answer | Evidence |
|----------|--------|----------|
| Which script produced target metrics (98.11%)? | `run_ablation.py` (root) | Lines 134-136: Full Hybrid config; Lines 138-147: print output |
| Which script produced thesis-ready metrics (96.23%)? | `notebooks/05B_bert_graph_fusion_fixed.py` | Lines 243-249: System D training; Lines 364-383: results JSON |
| What feature set was used? | MiniLM (384) + Propagation (4) + Graph (14) = **402 total** | `run_ablation.py` lines 128-136 |
| Which classifier was used? | LogisticRegression(C=1.0, max_iter=1000, random_state=42) | `run_ablation.py` lines 94, 107, 118 |
| What train/test split? | 80/20 stratified, random_state=42 | `run_ablation.py` lines 72-75 |
| Do serialized models exist? | **No** | `models/` directory is empty; no `.joblib`, `.pkl`, `.pth`, `.pt` found |
| Where to insert serialization? | `run_ablation.py` after line 136 OR `05B_bert_graph_fusion_fixed.py` after line 249 | See Section 5 above for exact locations and code |
| What two pipelines exist? | **Post-level** (05B, 102K posts) vs **Thread-level** (ablation, 5.8K threads) | Different files, different row counts, different metrics |