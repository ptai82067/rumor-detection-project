#!/usr/bin/env python
"""
BERT + Graph Feature Fusion Experiment Runner

This script executes the BERT + graph feature fusion experiments
for rumor detection. It can be run directly or used to execute
the notebook programmatically.
"""

import sys
import os
# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Define paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pheme_features.csv')
KG_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pheme_kg.ttl')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'project_brain_bundle', 'regression_pack', 'bert_graph_results.json')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

from utils.bert_feature_fusion import load_bert_model, extract_sentence_embeddings
from utils.graph_features import extract_all_graph_features, get_graph_feature_columns

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("=" * 80)
print("BERT + Graph Feature Fusion Experiment")
print("=" * 80)

# Load data
print("\n1. Loading frozen dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Dataset shape: {df.shape}")
print(f"   Total posts: {len(df):,}")

# Create train/test split
print("\n2. Creating frozen train/test split...")
y = df['label']
indices = np.arange(len(df))
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    indices, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
df_train = df.iloc[X_train_idx].copy()
df_test = df.iloc[X_test_idx].copy()
print(f"   Train: {len(X_train_idx):,}, Test: {len(X_test_idx):,}")

# Extract graph features
print("\n3. Extracting graph features...")
df_enriched = extract_all_graph_features(df, kg_path=KG_PATH)
graph_cols = get_graph_feature_columns()
print(f"   Graph features: {len(graph_cols)}")

# Add is_reply column
df['is_reply'] = (df['depth'] > 0).astype(int)
df_train['is_reply'] = (df_train['depth'] > 0).astype(int)
df_test['is_reply'] = (df_test['depth'] > 0).astype(int)
df_enriched['is_reply'] = (df_enriched['depth'] > 0).astype(int)

# Load BERT model
print("\n4. Loading BERT model...")
bert_model = load_bert_model()

# Baseline model
print("\n5. Running baseline model (TF-IDF + Propagation)...")
propagation_cols = ['is_reply', 'thread_size', 'children_count', 'depth']

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['text'])
X_test_tfidf = tfidf_vectorizer.transform(df_test['text'])

prop_scaler = StandardScaler()
X_train_prop_scaled = prop_scaler.fit_transform(df_train[propagation_cols].values)
X_test_prop_scaled = prop_scaler.transform(df_test[propagation_cols].values)

X_train_baseline = hstack([X_train_tfidf, X_train_prop_scaled])
X_test_baseline = hstack([X_test_tfidf, X_test_prop_scaled])

baseline_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
baseline_model.fit(X_train_baseline, y_train)
y_pred_baseline = baseline_model.predict(X_test_baseline)
y_pred_proba_baseline = baseline_model.predict_proba(X_test_baseline)[:, 1]

baseline_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_baseline),
    'Precision': precision_score(y_test, y_pred_baseline, pos_label=1),
    'Recall': recall_score(y_test, y_pred_baseline, pos_label=1),
    'F1-Score': f1_score(y_test, y_pred_baseline, pos_label=1),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_baseline),
    'MCC': matthews_corrcoef(y_test, y_pred_baseline),
}
print(f"   Recall: {baseline_metrics['Recall']:.4f}")

# Best graph model (from notebook 04)
print("\n6. Running best graph model (TF-IDF + Prop + Graph)...")
X_train_graph = df_enriched.iloc[X_train_idx][graph_cols].values
X_test_graph = df_enriched.iloc[X_test_idx][graph_cols].values
graph_scaler = StandardScaler()
X_train_graph_scaled = graph_scaler.fit_transform(X_train_graph)
X_test_graph_scaled = graph_scaler.transform(X_test_graph)

X_train_graph_model = hstack([X_train_baseline, X_train_graph_scaled])
X_test_graph_model = hstack([X_test_baseline, X_test_graph_scaled])

graph_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
graph_model.fit(X_train_graph_model, y_train)
y_pred_graph = graph_model.predict(X_test_graph_model)
y_pred_proba_graph = graph_model.predict_proba(X_test_graph_model)[:, 1]

graph_model_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_graph),
    'Precision': precision_score(y_test, y_pred_graph, pos_label=1),
    'Recall': recall_score(y_test, y_pred_graph, pos_label=1),
    'F1-Score': f1_score(y_test, y_pred_graph, pos_label=1),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_graph),
    'MCC': matthews_corrcoef(y_test, y_pred_graph),
}
print(f"   Recall: {graph_model_metrics['Recall']:.4f}")

# Extract BERT embeddings
print("\n7. Extracting BERT sentence embeddings...")
X_bert_train = extract_sentence_embeddings(df_train['text'].tolist(), bert_model, batch_size=32, show_progress=True)
X_bert_test = extract_sentence_embeddings(df_test['text'].tolist(), bert_model, batch_size=32, show_progress=True)
print(f"   BERT embedding shape: {X_bert_train.shape}")

# Experiment A: BERT Only
print("\n8. Running Experiment A: BERT Only...")
bert_only_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
bert_only_model.fit(X_bert_train, y_train)
y_pred_bert_only = bert_only_model.predict(X_bert_test)
y_pred_proba_bert_only = bert_only_model.predict_proba(X_bert_test)[:, 1]

bert_only_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_bert_only),
    'Precision': precision_score(y_test, y_pred_bert_only, pos_label=1),
    'Recall': recall_score(y_test, y_pred_bert_only, pos_label=1),
    'F1-Score': f1_score(y_test, y_pred_bert_only, pos_label=1),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_bert_only),
    'MCC': matthews_corrcoef(y_test, y_pred_bert_only),
}
print(f"   Recall: {bert_only_metrics['Recall']:.4f}")

# Experiment B: BERT + Propagation
print("\n9. Running Experiment B: BERT + Propagation...")
X_train_bert_prop = np.hstack([X_bert_train, X_train_prop_scaled])
X_test_bert_prop = np.hstack([X_bert_test, X_test_prop_scaled])

bert_prop_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
bert_prop_model.fit(X_train_bert_prop, y_train)
y_pred_bert_prop = bert_prop_model.predict(X_test_bert_prop)
y_pred_proba_bert_prop = bert_prop_model.predict_proba(X_test_bert_prop)[:, 1]

bert_prop_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_bert_prop),
    'Precision': precision_score(y_test, y_pred_bert_prop, pos_label=1),
    'Recall': recall_score(y_test, y_pred_bert_prop, pos_label=1),
    'F1-Score': f1_score(y_test, y_pred_bert_prop, pos_label=1),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_bert_prop),
    'MCC': matthews_corrcoef(y_test, y_pred_bert_prop),
}
print(f"   Recall: {bert_prop_metrics['Recall']:.4f}")

# Experiment C: BERT + Graph
print("\n10. Running Experiment C: BERT + Graph...")
X_train_bert_graph = np.hstack([X_bert_train, X_train_graph_scaled])
X_test_bert_graph = np.hstack([X_bert_test, X_test_graph_scaled])

bert_graph_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
bert_graph_model.fit(X_train_bert_graph, y_train)
y_pred_bert_graph = bert_graph_model.predict(X_test_bert_graph)
y_pred_proba_bert_graph = bert_graph_model.predict_proba(X_test_bert_graph)[:, 1]

bert_graph_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_bert_graph),
    'Precision': precision_score(y_test, y_pred_bert_graph, pos_label=1),
    'Recall': recall_score(y_test, y_pred_bert_graph, pos_label=1),
    'F1-Score': f1_score(y_test, y_pred_bert_graph, pos_label=1),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_bert_graph),
    'MCC': matthews_corrcoef(y_test, y_pred_bert_graph),
}
print(f"   Recall: {bert_graph_metrics['Recall']:.4f}")

# Experiment D: Full Fusion
print("\n11. Running Experiment D: Full Fusion (BERT + Prop + Graph)...")
X_train_full = np.hstack([X_bert_train, X_train_prop_scaled, X_train_graph_scaled])
X_test_full = np.hstack([X_bert_test, X_test_prop_scaled, X_test_graph_scaled])

full_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
full_model.fit(X_train_full, y_train)
y_pred_full = full_model.predict(X_test_full)
y_pred_proba_full = full_model.predict_proba(X_test_full)[:, 1]

full_model_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_full),
    'Precision': precision_score(y_test, y_pred_full, pos_label=1),
    'Recall': recall_score(y_test, y_pred_full, pos_label=1),
    'F1-Score': f1_score(y_test, y_pred_full, pos_label=1),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_full),
    'MCC': matthews_corrcoef(y_test, y_pred_full),
}
print(f"   Recall: {full_model_metrics['Recall']:.4f}")

# Summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

all_metrics = {
    'Baseline (TF-IDF + Prop)': baseline_metrics,
    'Best Graph (TF-IDF + Prop + Graph)': graph_model_metrics,
    'Exp A: BERT Only': bert_only_metrics,
    'Exp B: BERT + Prop': bert_prop_metrics,
    'Exp C: BERT + Graph': bert_graph_metrics,
    'Exp D: Full Fusion': full_model_metrics,
}

for name, metrics in all_metrics.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"   {metric:12s}: {value:.4f}")

# Find best recall
best_model_name = max(all_metrics, key=lambda x: all_metrics[x]['Recall'])
best_recall = all_metrics[best_model_name]['Recall']
print(f"\nBest model for Recall: {best_model_name}")
print(f"Best Recall: {best_recall:.4f}")
print(f"Target Recall (0.7775) achieved: {'YES' if best_recall > 0.7775 else 'NO'}")

# Save results
import json

bert_graph_results = {
    'experiment_metadata': {
        'dataset_size': len(df),
        'train_size': len(X_train_idx),
        'test_size': len(X_test_idx),
        'random_state': RANDOM_STATE,
        'bert_model': 'all-MiniLM-L6-v2',
        'bert_embedding_dim': 384,
        'graph_features': len(graph_cols),
        'propagation_features': len(propagation_cols),
        'classifier': 'LogisticRegression',
        'class_weight': 'balanced'
    },
    'baseline': baseline_metrics,
    'best_graph_model': graph_model_metrics,
    'ablation_study': {
        'exp_a_bert_only': bert_only_metrics,
        'exp_b_bert_prop': bert_prop_metrics,
        'exp_c_bert_graph': bert_graph_metrics,
        'exp_d_full_fusion': full_model_metrics
    },
    'success_criteria': {
        'target_recall': 0.7775,
        'best_recall_achieved': best_recall,
        'best_model': best_model_name,
        'recall_improvement': best_recall - baseline_metrics['Recall']
    }
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(bert_graph_results, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
print("\nExperiment completed successfully!")