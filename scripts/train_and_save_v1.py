#!/usr/bin/env python3
"""
V1 Serialization Pipeline -- Post-Level (05B Pipeline)
======================================================
Based on: notebooks/05B_bert_graph_fusion_fixed.py

Trains the Full Hybrid model (TF-IDF + MiniLM + Propagation + Graph)
and saves all artifacts for Streamlit deployment.

Output: models/v1/
  - model.joblib              Trained LogisticRegression
  - tfidf_vectorizer.joblib   Fitted TfidfVectorizer
  - scaler_graph.joblib       StandardScaler for graph features
  - scaler_propagation.joblib StandardScaler for propagation features
  - metadata.json             Experiment configuration and metrics
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from scipy.sparse import hstack
import joblib

# Add project root for utils imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.bert_feature_fusion import (
    load_bert_model,
    extract_sentence_embeddings,
)
from utils.graph_features import get_graph_feature_columns

OK = "[OK]"
FAIL = "[FAIL]"
INFO = "[INFO]"

print("=" * 80)
print("V1 SERIALIZATION PIPELINE (Post-Level -- 05B Pipeline)")
print("=" * 80)

# ============================================================
# CONSTANTS
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
INPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/pheme_features_with_graph.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "v1")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nConstants:")
print(f"  RANDOM_STATE: {RANDOM_STATE}")
print(f"  TEST_SIZE: {TEST_SIZE}")
print(f"  INPUT: {INPUT_PATH}")
print(f"  OUTPUT: {MODEL_DIR}")

# ============================================================
# STEP 1: Load and Validate Input Dataset
# ============================================================
print("\n" + "=" * 80)
print("STEP 1: Load and Validate Input Dataset")
print("=" * 80)

df = pd.read_csv(INPUT_PATH)
assert df.shape == (102440, 31), f"Expected (102440, 31), got {df.shape}"
print(f"  {OK} Input shape validated: {df.shape}")

graph_cols = get_graph_feature_columns()
missing = [c for c in graph_cols if c not in df.columns]
assert len(missing) == 0, f"Missing graph columns: {missing}"
print(f"  {OK} All 14 graph columns present")

# Validate non-zero topology signal
topology_cols = ['node_in_degree', 'pagerank_score', 'node_out_degree']
for col in topology_cols:
    non_zero = (df[col] != 0).sum()
    assert non_zero > 0, f"Topology column {col} has all zeros!"
    print(f"  {OK} {col}: {non_zero:,} non-zero values ({non_zero/len(df)*100:.1f}%)")

print(f"\nLabel distribution:")
print(f"  Non-Rumor: {(df['label'] == 0).sum():,} ({df['label'].mean():.1%})")
print(f"  Rumor: {(df['label'] == 1).sum():,} ({1 - df['label'].mean():.1%})")

# ============================================================
# STEP 2: Create Frozen Train/Test Split
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: Create Frozen Train/Test Split")
print("=" * 80)

y = df['label']
indices = np.arange(len(df))

X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    indices, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

df_train = df.iloc[X_train_idx].copy()
df_test = df.iloc[X_test_idx].copy()

print(f"\nTrain set: {len(X_train_idx):,} samples")
print(f"Test set: {len(X_test_idx):,} samples")
print(f"Train rumor ratio: {y_train.mean():.3f}")
print(f"Test rumor ratio: {y_test.mean():.3f}")

# ============================================================
# STEP 3: Extract Features for All Systems
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: Extract Features for All Systems")
print("=" * 80)

# --- 3a: TF-IDF Features ---
print(f"\n[3a] Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['text'])
X_test_tfidf = tfidf_vectorizer.transform(df_test['text'])
print(f"  TF-IDF dimension: {X_train_tfidf.shape[1]}")

# --- 3b: MiniLM BERT Embeddings ---
print(f"\n[3b] Loading MiniLM model and extracting embeddings...")
bert_model = load_bert_model()
print(f"  Model loaded: all-MiniLM-L6-v2 (384-dim)")

print("  Extracting train embeddings...")
X_bert_train = extract_sentence_embeddings(
    df_train['text'].tolist(), bert_model, batch_size=32, show_progress=True
)

print("  Extracting test embeddings...")
X_bert_test = extract_sentence_embeddings(
    df_test['text'].tolist(), bert_model, batch_size=32, show_progress=True
)

print(f"  BERT train shape: {X_bert_train.shape}")
print(f"  BERT test shape: {X_bert_test.shape}")

# --- 3c: Graph Features ---
print(f"\n[3c] Extracting graph features...")
graph_scaler = StandardScaler()
X_train_graph = graph_scaler.fit_transform(df_train[graph_cols].values)
X_test_graph = graph_scaler.transform(df_test[graph_cols].values)
print(f"  Graph features dimension: {len(graph_cols)}")

# --- 3d: Propagation Features ---
print(f"\n[3d] Extracting propagation features...")
propagation_cols = ['is_reply', 'thread_size', 'children_count', 'depth']
df['is_reply'] = (df['depth'] > 0).astype(int)
df_train['is_reply'] = (df_train['depth'] > 0).astype(int)
df_test['is_reply'] = (df_test['depth'] > 0).astype(int)

prop_scaler = StandardScaler()
X_train_prop = prop_scaler.fit_transform(df_train[propagation_cols].values)
X_test_prop = prop_scaler.transform(df_test[propagation_cols].values)
print(f"  Propagation features dimension: {len(propagation_cols)}")

# ============================================================
# STEP 4: Train and Evaluate All Systems
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: Train and Evaluate All Systems")
print("=" * 80)

def train_and_evaluate(X_train, X_test, y_train, y_test, name):
    """
    Train Logistic Regression and evaluate.
    Returns: (metrics_dict, confusion_matrix, fn, trained_model)
    """
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'Precision': round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        'F1-Score': round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        'F1 (Non-Rumor)': round(float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)), 4),
        'ROC-AUC': round(float(roc_auc_score(y_test, y_pred_proba)), 4),
        'MCC': round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    print(f"\n{'=' * 60}")
    print(f"SYSTEM: {name}")
    print(f"{'=' * 60}")
    for metric, value in metrics.items():
        print(f"  {metric:15s}: {value:.4f}")
    print(f"{'=' * 60}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Non-Rumor    Rumor")
    print(f"  Actual Non-Rumor   {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"         Rumor       {cm[1,0]:5d}     {cm[1,1]:5d}")
    print(f"\n  False Negatives (missed rumors): {fn}")
    print(f"  Rumor Recall: {tp/(tp+fn):.4f}")

    return metrics, cm.tolist(), fn, model

# --- System A: TF-IDF Baseline ---
print("\n" + "-" * 40)
print("SYSTEM A: TF-IDF + Propagation (Baseline)")
print("-" * 40)
X_train_A = hstack([X_train_tfidf, X_train_prop])
X_test_A = hstack([X_test_tfidf, X_test_prop])
metrics_A, cm_A, fn_A, model_A = train_and_evaluate(
    X_train_A, X_test_A, y_train, y_test, "A: TF-IDF + Propagation"
)

# --- System B: MiniLM Only ---
print("\n" + "-" * 40)
print("SYSTEM B: MiniLM Only")
print("-" * 40)
metrics_B, cm_B, fn_B, model_B = train_and_evaluate(
    X_bert_train, X_bert_test, y_train, y_test, "B: MiniLM Only"
)

# --- System C: MiniLM + Graph ---
print("\n" + "-" * 40)
print("SYSTEM C: MiniLM + Graph Features")
print("-" * 40)
X_train_C = np.hstack([X_bert_train, X_train_graph])
X_test_C = np.hstack([X_bert_test, X_test_graph])
metrics_C, cm_C, fn_C, model_C = train_and_evaluate(
    X_train_C, X_test_C, y_train, y_test, "C: MiniLM + Graph"
)

# --- System D: TF-IDF + MiniLM + Graph Hybrid (THE DEPLOYMENT MODEL) ---
print("\n" + "-" * 40)
print("SYSTEM D: TF-IDF + MiniLM + Graph Hybrid")
print("-" * 40)
X_train_D = np.hstack([X_train_tfidf.toarray(), X_bert_train, X_train_graph])
X_test_D = np.hstack([X_test_tfidf.toarray(), X_bert_test, X_test_graph])
metrics_D, cm_D, fn_D, model_D = train_and_evaluate(
    X_train_D, X_test_D, y_train, y_test, "D: TF-IDF + MiniLM + Graph Hybrid"
)

# ============================================================
# STEP 5: Save All Artifacts
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: Saving V1 Artifacts")
print("=" * 80)

# 5a: Save trained model
model_path = os.path.join(MODEL_DIR, "model.joblib")
joblib.dump(model_D, model_path)
print(f"  {OK} Model saved: {model_path}")

# 5b: Save TF-IDF vectorizer
vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
joblib.dump(tfidf_vectorizer, vec_path)
print(f"  {OK} TF-IDF vectorizer saved: {vec_path}")

# 5c: Save graph scaler
scaler_g_path = os.path.join(MODEL_DIR, "scaler_graph.joblib")
joblib.dump(graph_scaler, scaler_g_path)
print(f"  {OK} Graph scaler saved: {scaler_g_path}")

# 5d: Save propagation scaler
scaler_p_path = os.path.join(MODEL_DIR, "scaler_propagation.joblib")
joblib.dump(prop_scaler, scaler_p_path)
print(f"  {OK} Propagation scaler saved: {scaler_p_path}")

# 5e: Save metadata
metadata = {
    "pipeline": "V1 -- Post-Level (05B Pipeline)",
    "training_script": "scripts/train_and_save_v1.py",
    "based_on": "notebooks/05B_bert_graph_fusion_fixed.py",
    "dataset": INPUT_PATH,
    "dataset_shape": list(df.shape),
    "train_size": int(len(X_train_idx)),
    "test_size": int(len(X_test_idx)),
    "random_state": RANDOM_STATE,
    "classifier": "LogisticRegression",
    "classifier_params": {
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "class_weight": "balanced"
    },
    "features": {
        "tfidf": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "stop_words": "english"
        },
        "minilm": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "level": "post-level (full text)"
        },
        "propagation": {
            "columns": propagation_cols,
            "dimension": len(propagation_cols)
        },
        "graph": {
            "columns": graph_cols,
            "dimension": len(graph_cols)
        }
    },
    "feature_dimensions": {
        "tfidf": int(X_train_tfidf.shape[1]),
        "minilm": 384,
        "propagation": len(propagation_cols),
        "graph": len(graph_cols),
        "total_hybrid": int(X_train_D.shape[1])
    },
    "metrics": {
        "system_A_TFIDF_Propagation": {
            **metrics_A,
            "false_negatives": fn_A,
            "confusion_matrix": cm_A
        },
        "system_B_MiniLM_only": {
            **metrics_B,
            "false_negatives": fn_B,
            "confusion_matrix": cm_B
        },
        "system_C_MiniLM_Graph": {
            **metrics_C,
            "false_negatives": fn_C,
            "confusion_matrix": cm_C
        },
        "system_D_Hybrid": {
            **metrics_D,
            "false_negatives": fn_D,
            "confusion_matrix": cm_D
        }
    },
    "artifacts": [
        "model.joblib -- Trained LogisticRegression (System D: Hybrid)",
        "tfidf_vectorizer.joblib -- Fitted TfidfVectorizer",
        "scaler_graph.joblib -- StandardScaler for 14 graph features",
        "scaler_propagation.joblib -- StandardScaler for 4 propagation features",
        "metadata.json -- This configuration file"
    ],
    "generation_date": "2026-06-07"
}

metadata_path = os.path.join(MODEL_DIR, "metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  {OK} Metadata saved: {metadata_path}")

# ============================================================
# STEP 6: Summary
# ============================================================
print("\n" + "=" * 80)
print("V1 SERIALIZATION COMPLETE")
print("=" * 80)
print(f"\nBest Model: System D -- TF-IDF + MiniLM + Graph Hybrid")
print(f"  Accuracy:  {metrics_D['Accuracy']:.4f}")
print(f"  Precision: {metrics_D['Precision']:.4f}")
print(f"  Recall:    {metrics_D['Recall']:.4f}")
print(f"  F1-Score:  {metrics_D['F1-Score']:.4f}")
print(f"  ROC-AUC:   {metrics_D['ROC-AUC']:.4f}")
print(f"  MCC:       {metrics_D['MCC']:.4f}")
print(f"  FN:        {fn_D}")
print(f"\nAll artifacts saved to: {MODEL_DIR}")
print("=" * 80)