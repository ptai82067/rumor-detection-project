#!/usr/bin/env python3
"""
V2 Serialization Pipeline -- Thread-Level (run_ablation.py Pipeline)
====================================================================
Based on: run_ablation.py

Reproduces the Full Hybrid configuration (MiniLM + Propagation + Graph)
at thread level and saves all artifacts for Streamlit deployment.

Output: models/v2/
  - model.joblib       Trained LogisticRegression (Full Hybrid)
  - scaler.joblib      StandardScaler for combined features
  - metadata.json      Experiment configuration and metrics
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import csv
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sentence_transformers import SentenceTransformer
import joblib

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "v2")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "ablation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

OK = "[OK]"
FAIL = "[FAIL]"

print("=" * 80)
print("V2 SERIALIZATION PIPELINE (Thread-Level -- Ablation Pipeline)")
print("=" * 80)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n" + "=" * 80)
print("STEP 1: Load Data")
print("=" * 80)

base_df = pd.read_csv(os.path.join(DATA_DIR, "pheme_features.csv"), dtype={'reply_to': str})
graph_df = pd.read_csv(os.path.join(DATA_DIR, "graph_features_v2.csv"))
print(f"  Base features: {base_df.shape}")
print(f"  Graph features v2: {graph_df.shape}")

# Build thread-level dataframe (same as run_ablation.py)
print("\nBuilding thread-level dataframe...")
thread_data = []
for tid, group in base_df.groupby('thread_id'):
    source = group[group['depth'] == 0].iloc[0] if len(group[group['depth'] == 0]) > 0 else group.iloc[0]
    thread_data.append({
        'thread_id': int(tid),
        'label': int(source['label']),
        'text': ' '.join(group['text'].dropna().astype(str).tolist()),
        'source_text': str(source['text'])
    })
thread_df = pd.DataFrame(thread_data)
thread_df = thread_df.merge(graph_df, on='thread_id', how='left')
print(f"  Threads: {thread_df.shape}, Nulls: {thread_df.isnull().sum().sum()}")

# ============================================================
# STEP 2: BUILD PROPAGATION FEATURES
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: Build Propagation Features")
print("=" * 80)

prop_cols = ['thread_size', 'max_depth', 'avg_depth_prop', 'reply_rate']
prop_data = base_df.groupby('thread_id').agg(
    thread_size=('thread_size', 'first'),
    max_depth=('max_depth', 'first')
).reset_index()

avg_d = base_df.groupby('thread_id')['depth'].mean().reset_index()
avg_d.columns = ['thread_id', 'avg_depth_prop']

cnt = base_df.groupby('thread_id').size().reset_index()
cnt.columns = ['thread_id', 'total_posts']

prop_data = prop_data.merge(avg_d, on='thread_id').merge(cnt, on='thread_id')
prop_data['reply_rate'] = prop_data['total_posts'] / (prop_data['thread_size'] + 1)
thread_df = thread_df.merge(prop_data[['thread_id'] + prop_cols], on='thread_id', how='left')

print(f"  Propagation columns: {prop_cols}")
print(f"  Propagation features shape: {thread_df[prop_cols].shape}")

# ============================================================
# STEP 3: GENERATE MINILM EMBEDDINGS
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: Generate MiniLM Embeddings (Source Text Only)")
print("=" * 80)

print("  Loading SentenceTransformer('all-MiniLM-L6-v2')...")
model = SentenceTransformer('all-MiniLM-L6-v2')
source_texts = thread_df['source_text'].fillna('').tolist()
print(f"  Encoding {len(source_texts)} source texts...")
embeddings = model.encode(
    source_texts, show_progress_bar=True, batch_size=64
)
print(f"  Embeddings shape: {embeddings.shape}")

# ============================================================
# STEP 4: CREATE TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: Create Train/Test Split")
print("=" * 80)

y = thread_df['label'].values
train_idx, test_idx = train_test_split(
    np.arange(len(thread_df)), test_size=0.2, random_state=42, stratify=y
)
y_train, y_test = y[train_idx], y[test_idx]
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

# ============================================================
# STEP 5: BUILD FEATURE MATRICES
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: Build Feature Matrices")
print("=" * 80)

graph_cols = [
    'thread_depth', 'num_nodes', 'num_edges', 'avg_branching_factor',
    'max_branching_factor', 'source_reply_count', 'leaf_ratio', 'avg_depth',
    'source_pagerank', 'avg_pagerank', 'source_centrality', 'avg_centrality',
    'user_rumor_ratio', 'unique_users'
]

X_prop = thread_df[prop_cols].fillna(0).values
X_graph = thread_df[graph_cols].fillna(0).values

print(f"  Propagation features: {X_prop.shape[1]}")
print(f"  Graph features:       {X_graph.shape[1]}")
print(f"  MiniLM embeddings:    {embeddings.shape[1]}")

# Build the Full Hybrid feature matrix: MiniLM + Propagation + Graph
X_full = np.hstack([embeddings, X_prop, X_graph])
print(f"  Full Hybrid dimension: {X_full.shape[1]}")

# ============================================================
# STEP 6: SCALE AND TRAIN
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: Scale Features and Train Full Hybrid Model")
print("=" * 80)

# StandardScaler fit on train, transform both train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_full[train_idx])
X_test_scaled = scaler.transform(X_full[test_idx])

# Train LogisticRegression (no class_weight, matching run_ablation.py)
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
accuracy = float(accuracy_score(y_test, y_pred))
precision_val = float(precision_score(y_test, y_pred, zero_division=0))
recall_val = float(recall_score(y_test, y_pred, zero_division=0))
f1_val = float(f1_score(y_test, y_pred, zero_division=0))
fn = int(cm[1, 0])

print(f"\n{'=' * 60}")
print("SYSTEM: Full Hybrid (MiniLM + Propagation + Graph)")
print(f"{'=' * 60}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision_val:.4f}")
print(f"  Recall:    {recall_val:.4f}")
print(f"  F1-Score:  {f1_val:.4f}")
print(f"  FN:        {fn}")
print(f"{'=' * 60}")
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"                Non-Rumor    Rumor")
print(f"  Actual Non-Rumor   {cm[0,0]:5d}     {cm[0,1]:5d}")
print(f"         Rumor       {cm[1,0]:5d}     {cm[1,1]:5d}")

# ============================================================
# STEP 7: SAVE ARTIFACTS
# ============================================================
print("\n" + "=" * 80)
print("STEP 7: Save V2 Artifacts")
print("=" * 80)

# 7a: Save trained model
model_path = os.path.join(MODEL_DIR, "model.joblib")
joblib.dump(clf, model_path)
print(f"  {OK} Model saved: {model_path}")

# 7b: Save scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"  {OK} Scaler saved: {scaler_path}")

# 7c: Save metadata
metadata = {
    "pipeline": "V2 -- Thread-Level (Ablation Pipeline)",
    "training_script": "scripts/train_and_save_v2.py",
    "based_on": "run_ablation.py",
    "dataset": {
        "base_features": "data/processed/pheme_features.csv",
        "graph_features_v2": "data/processed/graph_features_v2.csv"
    },
    "dataset_shape": {
        "posts": list(base_df.shape),
        "threads": list(thread_df.shape)
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
        "stratified": True,
        "train_size": int(len(train_idx)),
        "test_size_count": int(len(test_idx))
    },
    "classifier": "LogisticRegression",
    "classifier_params": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": None
    },
    "features": {
        "minilm": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "level": "thread-level (source text only)"
        },
        "propagation": {
            "columns": prop_cols,
            "count": len(prop_cols)
        },
        "graph": {
            "columns": graph_cols,
            "count": len(graph_cols)
        }
    },
    "feature_dimensions": {
        "minilm": 384,
        "propagation": len(prop_cols),
        "graph": len(graph_cols),
        "total_full_hybrid": int(X_full.shape[1])
    },
    "metrics": {
        "model": "Full Hybrid (MiniLM + Propagation + Graph)",
        "accuracy": round(accuracy, 4),
        "precision": round(precision_val, 4),
        "recall": round(recall_val, 4),
        "f1_score": round(f1_val, 4),
        "false_negatives": fn,
        "confusion_matrix": cm.tolist()
    },
    "ablation_reference": {
        "script": "run_ablation.py",
        "expected_accuracy": 0.9811,
        "expected_recall": 0.9646,
        "expected_f1": 0.9719
    },
    "artifacts": [
        "model.joblib -- Trained LogisticRegression (Full Hybrid)",
        "scaler.joblib -- StandardScaler for combined features (384+4+14=402)",
        "metadata.json -- This configuration file"
    ],
    "generation_date": "2026-06-07"
}

metadata_path = os.path.join(MODEL_DIR, "metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  {OK} Metadata saved: {metadata_path}")

# ============================================================
# STEP 8: SAVE ABLATION TABLE (for reference)
# ============================================================
print("\n" + "=" * 80)
print("STEP 8: Save Ablation Results (All 5 Configs for Reference)")
print("=" * 80)

# Run all 5 configs for completeness (matching run_ablation.py)
results = {}
configs = ['TF-IDF', 'MiniLM', 'MiniLM+Prop', 'MiniLM+Graph', 'Full_Hybrid']
dnames = ['TF-IDF', 'MiniLM', '+Propagation', '+Graph(KG v2)', 'Full Hybrid']
fnames = ['10K', '384', '384+4', '384+14', '384+4+14']

# Config 1: TF-IDF
print("\nConfig 1: TF-IDF (10K features)...")
tfidf_vec = TfidfVectorizer(max_features=10000, stop_words='english',
                             ngram_range=(1, 2), min_df=2, max_df=0.95)
X_tfidf = tfidf_vec.fit_transform(thread_df['text'].fillna(''))
clf_tfidf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf_tfidf.fit(X_tfidf[train_idx], y_train)
yp = clf_tfidf.predict(X_tfidf[test_idx])
cm_tfidf = confusion_matrix(y_test, yp)
results['TF-IDF'] = {
    'Acc': round(accuracy_score(y_test, yp), 4),
    'Prec': round(precision_score(y_test, yp, zero_division=0), 4),
    'Recall': round(recall_score(y_test, yp, zero_division=0), 4),
    'F1': round(f1_score(y_test, yp, zero_division=0), 4),
    'FN': int(cm_tfidf[1][0])
}

# Config 2: MiniLM Only
print("Config 2: MiniLM only...")
ss = StandardScaler()
X_t = ss.fit_transform(embeddings[train_idx])
X_e = ss.transform(embeddings[test_idx])
clf_ml = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf_ml.fit(X_t, y_train)
yp = clf_ml.predict(X_e)
cm_ml = confusion_matrix(y_test, yp)
results['MiniLM'] = {
    'Acc': round(accuracy_score(y_test, yp), 4),
    'Prec': round(precision_score(y_test, yp, zero_division=0), 4),
    'Recall': round(recall_score(y_test, yp, zero_division=0), 4),
    'F1': round(f1_score(y_test, yp, zero_division=0), 4),
    'FN': int(cm_ml[1][0])
}

def run_config(X_data, name):
    ss = StandardScaler()
    X_t = ss.fit_transform(X_data[train_idx])
    X_e = ss.transform(X_data[test_idx])
    clf_c = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_c.fit(X_t, y_train)
    yp = clf_c.predict(X_e)
    cm_c = confusion_matrix(y_test, yp)
    results[name] = {
        'Acc': round(accuracy_score(y_test, yp), 4),
        'Prec': round(precision_score(y_test, yp, zero_division=0), 4),
        'Recall': round(recall_score(y_test, yp, zero_division=0), 4),
        'F1': round(f1_score(y_test, yp, zero_division=0), 4),
        'FN': int(cm_c[1][0])
    }

# Config 3: MiniLM + Prop
print("Config 3: MiniLM + Propagation...")
run_config(np.hstack([embeddings, X_prop]), 'MiniLM+Prop')

# Config 4: MiniLM + Graph
print("Config 4: MiniLM + Graph...")
run_config(np.hstack([embeddings, X_graph]), 'MiniLM+Graph')

# Config 5: Full Hybrid (already done above)
results['Full_Hybrid'] = {
    'Acc': round(accuracy, 4),
    'Prec': round(precision_val, 4),
    'Recall': round(recall_val, 4),
    'F1': round(f1_val, 4),
    'FN': int(fn)
}

# Print results
print("\n" + "=" * 90)
print("ABLATION STUDY RESULTS")
print("=" * 90)
print(f"{'Config':<20} {'Feat':<10} {'Acc':<9} {'Prec':<9} {'Recall':<9} {'F1':<9} {'FN':<7}")
print("-" * 75)
for i, c in enumerate(configs):
    r = results[c]
    print(f"{dnames[i]:<20} {fnames[i]:<10} {r['Acc']:.4f}    {r['Prec']:.4f}    "
          f"{r['Recall']:.4f}    {r['F1']:.4f}    {r['FN']}")

# Save ablation table
base_recall = results['TF-IDF']['Recall']
base_fn = results['TF-IDF']['FN']
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, 'ablation_table.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Config', 'Features', 'Acc', 'Prec', 'Recall', 'F1', 'FN',
                'Delta_Recall_vs_TFIDF', 'Delta_FN_vs_TFIDF'])
    for i, c in enumerate(configs):
        r = results[c]
        star = ' *' if c == 'Full_Hybrid' else ''
        w.writerow([dnames[i] + star, fnames[i], r['Acc'], r['Prec'],
                    r['Recall'], r['F1'], r['FN'],
                    round(r['Recall'] - base_recall, 4),
                    int(base_fn - r['FN'])])

print(f"  {OK} Ablation table saved to: {OUT_DIR}/ablation_table.csv")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("V2 SERIALIZATION COMPLETE")
print("=" * 80)
print(f"\nBest Model: Full Hybrid (MiniLM + Propagation + Graph)")
print(f"  Accuracy:  {accuracy:.4f}  (expected: 0.9811)")
print(f"  Precision: {precision_val:.4f}  (expected: 0.9794)")
print(f"  Recall:    {recall_val:.4f}  (expected: 0.9646)")
print(f"  F1-Score:  {f1_val:.4f}  (expected: 0.9719)")
print(f"  FN:        {fn}  (expected: 14)")
print(f"\nAll artifacts saved to: {MODEL_DIR}")
print("=" * 80)