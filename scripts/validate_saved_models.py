#!/usr/bin/env python3
"""
Model Validation Script — V1 and V2 Serialized Artifacts
=========================================================
Purpose:
  1. Load V1 artifacts (model, vectorizer, scalers)
  2. Load V2 artifacts (model, scaler)
  3. Run sample predictions on test data
  4. Verify feature dimensions match training expectations
  5. Generate reports/model_validation_report.md

Usage:
  python scripts/validate_saved_models.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import joblib

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.bert_feature_fusion import load_bert_model, extract_sentence_embeddings
from utils.graph_features import get_graph_feature_columns

# Paths
V1_DIR = os.path.join(PROJECT_ROOT, "models", "v1")
V2_DIR = os.path.join(PROJECT_ROOT, "models", "v2")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 80)
print("MODEL VALIDATION — V1 and V2 Serialized Artifacts")
print("=" * 80)

# ============================================================
# Validation Results Storage
# ============================================================
validation_results = {
    "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "v1": {"status": "FAIL", "details": {}, "checks": []},
    "v2": {"status": "FAIL", "details": {}, "checks": []},
    "summary": {
        "v1_status": "NOT RUN",
        "v2_status": "NOT RUN",
        "overall": "FAIL"
    }
}

# ============================================================
# V1 VALIDATION
# ============================================================
print("\n" + "=" * 80)
print("VALIDATING V1 ARTIFACTS")
print("=" * 80)

v1_artifacts = {
    "model.joblib": os.path.join(V1_DIR, "model.joblib"),
    "tfidf_vectorizer.joblib": os.path.join(V1_DIR, "tfidf_vectorizer.joblib"),
    "scaler_graph.joblib": os.path.join(V1_DIR, "scaler_graph.joblib"),
    "scaler_propagation.joblib": os.path.join(V1_DIR, "scaler_propagation.joblib"),
    "metadata.json": os.path.join(V1_DIR, "metadata.json"),
}

v1_all_pass = True
v1_objects = {}

for name, path in v1_artifacts.items():
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    check = {
        "artifact": name,
        "path": path,
        "exists": exists,
        "size_kb": round(size_kb, 2)
    }
    
    if not exists:
        check["status"] = "FAIL — File not found"
        v1_all_pass = False
        validation_results["v1"]["checks"].append(check)
        print(f"  ❌ {name}: NOT FOUND at {path}")
        continue
    
    try:
        if name == "metadata.json":
            with open(path, 'r') as f:
                obj = json.load(f)
            v1_objects["metadata"] = obj
            check["status"] = "PASS"
            check["pipeline"] = obj.get("pipeline", "unknown")
            check["metrics"] = obj.get("metrics", {}).get("system_D_Hybrid", {})
        else:
            obj = joblib.load(path)
            v1_objects[name.replace(".joblib", "").replace("_", " ").strip()] = obj
            check["status"] = "PASS"
            
            # Type-specific checks
            if "model" in name:
                from sklearn.linear_model import LogisticRegression
                if isinstance(obj, LogisticRegression):
                    check["type"] = "LogisticRegression"
                    check["n_classes"] = len(obj.classes_) if hasattr(obj, "classes_") else "unknown"
                    check["coef_shape"] = list(obj.coef_.shape) if hasattr(obj, "coef_") else "untrained"
                else:
                    check["type"] = type(obj).__name__
            elif "tfidf" in name:
                from sklearn.feature_extraction.text import TfidfVectorizer
                if isinstance(obj, TfidfVectorizer):
                    vocab_size = len(obj.vocabulary_) if hasattr(obj, "vocabulary_") else 0
                    check["type"] = "TfidfVectorizer"
                    check["vocabulary_size"] = vocab_size
                else:
                    check["type"] = type(obj).__name__
            elif "scaler" in name:
                from sklearn.preprocessing import StandardScaler
                if isinstance(obj, StandardScaler):
                    check["type"] = "StandardScaler"
                    check["mean_shape"] = list(obj.mean_.shape) if hasattr(obj, "mean_") else "unfitted"
                else:
                    check["type"] = type(obj).__name__
    
        print(f"  ✅ {name}: LOADED ({check.get('type', 'OK')})")
        
    except Exception as e:
        check["status"] = f"FAIL — {str(e)}"
        v1_all_pass = False
        print(f"  ❌ {name}: LOAD FAILED — {str(e)}")
    
    validation_results["v1"]["checks"].append(check)

# V1 cross-validation checks
if v1_all_pass:
    print("\n  --- V1 Cross-Validation Checks ---")
    
    # Check feature dimension consistency
    meta = v1_objects.get("metadata", {})
    feature_dims = meta.get("feature_dimensions", {})
    
    if feature_dims:
        expected_tfidf = feature_dims.get("tfidf", 0)
        expected_graph = feature_dims.get("graph", 0)
        expected_prop = feature_dims.get("propagation", 0)
        expected_total = feature_dims.get("total_hybrid", 0)
        
        print(f"  ✓ TF-IDF dimension: {expected_tfidf}")
        print(f"  ✓ Graph features: {expected_graph}")
        print(f"  ✓ Propagation features: {expected_prop}")
        print(f"  ✓ Total hybrid dimension: {expected_total}")
        
        # Check scaler dimensions
        if "scaler_graph" in v1_objects:
            mean_shape = v1_objects["scaler_graph"].mean_.shape[0]
            match = mean_shape == expected_graph
            print(f"  {'✓' if match else '✗'} Graph scaler dim: {mean_shape} == expected {expected_graph}: {match}")
            if not match: v1_all_pass = False
        
        if "scaler_propagation" in v1_objects:
            mean_shape = v1_objects["scaler_propagation"].mean_.shape[0]
            match = mean_shape == expected_prop
            print(f"  {'✓' if match else '✗'} Propagation scaler dim: {mean_shape} == expected {expected_prop}: {match}")
            if not match: v1_all_pass = False
        
        # Check model coefficients
        if v1_objects.get("model") is not None:
            coef_shape = v1_objects["model"].coef_.shape
            expected_coef = (1, expected_total)
            match = list(coef_shape) == list(expected_coef)
            print(f"  {'✓' if match else '✗'} Model coef shape: {list(coef_shape)} == expected {list(expected_coef)}: {match}")
            if not match: v1_all_pass = False
    else:
        print("  ⚠️  No feature dimensions in metadata.json")
        v1_all_pass = False

    validation_results["v1"]["details"] = {
        "feature_dimensions": feature_dims,
        "metrics": meta.get("metrics", {}).get("system_D_Hybrid", {})
    }

validation_results["v1"]["status"] = "PASS" if v1_all_pass else "FAIL"
print(f"\n  V1 Overall: {'✅ PASS' if v1_all_pass else '❌ FAIL'}")

# ============================================================
# V2 VALIDATION
# ============================================================
print("\n" + "=" * 80)
print("VALIDATING V2 ARTIFACTS")
print("=" * 80)

v2_artifacts = {
    "model.joblib": os.path.join(V2_DIR, "model.joblib"),
    "scaler.joblib": os.path.join(V2_DIR, "scaler.joblib"),
    "metadata.json": os.path.join(V2_DIR, "metadata.json"),
}

v2_all_pass = True
v2_objects = {}

for name, path in v2_artifacts.items():
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    check = {
        "artifact": name,
        "path": path,
        "exists": exists,
        "size_kb": round(size_kb, 2)
    }
    
    if not exists:
        check["status"] = "FAIL — File not found"
        v2_all_pass = False
        validation_results["v2"]["checks"].append(check)
        print(f"  ❌ {name}: NOT FOUND at {path}")
        continue
    
    try:
        if name == "metadata.json":
            with open(path, 'r') as f:
                obj = json.load(f)
            v2_objects["metadata"] = obj
            check["status"] = "PASS"
            check["pipeline"] = obj.get("pipeline", "unknown")
            check["metrics"] = obj.get("metrics", {})
        else:
            obj = joblib.load(path)
            v2_objects[name.replace(".joblib", "")] = obj
            check["status"] = "PASS"
            
            if "model" in name:
                from sklearn.linear_model import LogisticRegression
                if isinstance(obj, LogisticRegression):
                    check["type"] = "LogisticRegression"
                    check["n_classes"] = len(obj.classes_) if hasattr(obj, "classes_") else "unknown"
                    check["coef_shape"] = list(obj.coef_.shape) if hasattr(obj, "coef_") else "untrained"
                else:
                    check["type"] = type(obj).__name__
            elif "scaler" in name:
                from sklearn.preprocessing import StandardScaler
                if isinstance(obj, StandardScaler):
                    check["type"] = "StandardScaler"
                    check["mean_shape"] = list(obj.mean_.shape) if hasattr(obj, "mean_") else "unfitted"
                else:
                    check["type"] = type(obj).__name__
                    
        print(f"  ✅ {name}: LOADED ({check.get('type', 'OK')})")
        
    except Exception as e:
        check["status"] = f"FAIL — {str(e)}"
        v2_all_pass = False
        print(f"  ❌ {name}: LOAD FAILED — {str(e)}")
    
    validation_results["v2"]["checks"].append(check)

# V2 cross-validation checks
if v2_all_pass:
    print("\n  --- V2 Cross-Validation Checks ---")
    
    meta = v2_objects.get("metadata", {})
    feature_dims = meta.get("feature_dimensions", {})
    expected_total = feature_dims.get("total_full_hybrid", 0)
    
    if feature_dims:
        print(f"  ✓ MiniLM dimension: {feature_dims.get('minilm', 0)}")
        print(f"  ✓ Propagation features: {feature_dims.get('propagation', 0)}")
        print(f"  ✓ Graph features: {feature_dims.get('graph', 0)}")
        print(f"  ✓ Total hybrid dimension: {expected_total}")
        
        # Check scaler dimension
        if "scaler" in v2_objects:
            mean_shape = v2_objects["scaler"].mean_.shape[0]
            match = mean_shape == expected_total
            print(f"  {'✓' if match else '✗'} Scaler dim: {mean_shape} == expected {expected_total}: {match}")
            if not match: v2_all_pass = False
        
        # Check model coefficients
        if v2_objects.get("model") is not None:
            coef_shape = v2_objects["model"].coef_.shape
            expected_coef = (1, expected_total)
            match = list(coef_shape) == list(expected_coef)
            print(f"  {'✓' if match else '✗'} Model coef shape: {list(coef_shape)} == expected {list(expected_coef)}: {match}")
            if not match: v2_all_pass = False
    else:
        print("  ⚠️  No feature dimensions in metadata.json")
        v2_all_pass = False
    
    validation_results["v2"]["details"] = {
        "feature_dimensions": feature_dims,
        "metrics": meta.get("metrics", {})
    }

validation_results["v2"]["status"] = "PASS" if v2_all_pass else "FAIL"
print(f"\n  V2 Overall: {'✅ PASS' if v2_all_pass else '❌ FAIL'}")

# ============================================================
# RUN SAMPLE PREDICTIONS (if both pipelines loaded)
# ============================================================
print("\n" + "=" * 80)
print("SAMPLE PREDICTION TEST")
print("=" * 80)

sample_results = {"v1": {}, "v2": {}}

if v1_all_pass:
    try:
        print("\n--- V1 Sample Prediction ---")
        
        # Load a small sample from the test set
        input_path = os.path.join(PROJECT_ROOT, "data/processed/pheme_features_with_graph.csv")
        df = pd.read_csv(input_path).head(5)
        
        model_v1 = v1_objects["model"]
        tfidf_vec = v1_objects["tfidf_vectorizer"]
        scaler_g = v1_objects["scaler_graph"]
        scaler_p = v1_objects["scaler_propagation"]
        graph_cols = get_graph_feature_columns()
        prop_cols = ['is_reply', 'thread_size', 'children_count', 'depth']
        
        # Prepare features for sample
        X_tfidf_sample = tfidf_vec.transform(df['text'])
        df['is_reply'] = (df['depth'] > 0).astype(int)
        X_prop_sample = scaler_p.transform(df[prop_cols].values)
        X_graph_sample = scaler_g.transform(df[graph_cols].values)
        
        # Load BERT model for embedding extraction
        bert_model = load_bert_model()
        X_bert_sample = extract_sentence_embeddings(df['text'].tolist(), bert_model)
        
        # Concatenate (same as System D)
        X_sample = np.hstack([X_tfidf_sample.toarray(), X_bert_sample, X_graph_sample])
        
        # Predict
        y_pred = model_v1.predict(X_sample)
        y_proba = model_v1.predict_proba(X_sample)
        
        for i in range(len(df)):
            label = "Rumor" if y_pred[i] == 1 else "Non-Rumor"
            confidence = y_proba[i][1] if y_pred[i] == 1 else y_proba[i][0]
            print(f"  Sample {i+1}: Predicted={label}, Confidence={confidence:.4f}")
        
        sample_results["v1"] = {
            "status": "PASS",
            "n_samples": len(df),
            "predictions": [{"index": int(i), "prediction": int(y_pred[i]), 
                            "confidence": float(max(y_proba[i]))} for i in range(len(df))]
        }
        print(f"  ✅ V1 prediction successful on {len(df)} samples")
        
    except Exception as e:
        sample_results["v1"] = {"status": f"FAIL — {str(e)}"}
        print(f"  ❌ V1 prediction failed: {str(e)}")

if v2_all_pass:
    try:
        print("\n--- V2 Sample Prediction ---")
        
        # Load a small sample from the thread-level dataset
        data_dir = os.path.join(PROJECT_ROOT, "data/processed")
        base_df = pd.read_csv(os.path.join(data_dir, "pheme_features.csv"), dtype={'reply_to': str})
        graph_df = pd.read_csv(os.path.join(data_dir, "graph_features_v2.csv"))
        
        # Build a single thread sample
        tid_sample = graph_df['thread_id'].iloc[0]
        graph_sample = graph_df[graph_df['thread_id'] == tid_sample]
        
        # Get source text
        thread_posts = base_df[base_df['thread_id'] == tid_sample]
        source = thread_posts[thread_posts['depth'] == 0].iloc[0] if len(thread_posts[thread_posts['depth'] == 0]) > 0 else thread_posts.iloc[0]
        source_text = str(source['text'])
        
        # Build propagation features
        thread_size = int(thread_posts['thread_size'].iloc[0])
        max_depth = int(thread_posts['max_depth'].iloc[0])
        avg_depth = float(thread_posts['depth'].mean())
        reply_rate = len(thread_posts) / (thread_size + 1)
        prop_sample = np.array([[thread_size, max_depth, avg_depth, reply_rate]])
        
        # Build graph features
        graph_feature_cols = [
            'thread_depth', 'num_nodes', 'num_edges', 'avg_branching_factor',
            'max_branching_factor', 'source_reply_count', 'leaf_ratio', 'avg_depth',
            'source_pagerank', 'avg_pagerank', 'source_centrality', 'avg_centrality',
            'user_rumor_ratio', 'unique_users'
        ]
        graph_sample_values = graph_sample[graph_feature_cols].fillna(0).values
        
        # Generate MiniLM embedding
        model_v2 = v2_objects["model"]
        scaler_v2 = v2_objects["scaler"]
        
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        emb_sample = st_model.encode([source_text])
        
        # Concatenate all features
        X_full_sample = np.hstack([emb_sample, prop_sample, graph_sample_values])
        X_scaled = scaler_v2.transform(X_full_sample)
        
        # Predict
        y_pred = model_v2.predict(X_scaled)
        y_proba = model_v2.predict_proba(X_scaled)
        
        label = "Rumor" if y_pred[0] == 1 else "Non-Rumor"
        confidence = y_proba[0][1] if y_pred[0] == 1 else y_proba[0][0]
        print(f"  Thread {tid_sample}: Predicted={label}, Confidence={confidence:.4f}")
        print(f"  Source text: {source_text[:80]}...")
        
        sample_results["v2"] = {
            "status": "PASS",
            "thread_id": int(tid_sample),
            "prediction": int(y_pred[0]),
            "confidence": float(confidence),
            "label": label
        }
        print(f"  ✅ V2 prediction successful")
        
    except Exception as e:
        sample_results["v2"] = {"status": f"FAIL — {str(e)}"}
        print(f"  ❌ V2 prediction failed: {str(e)}")

# ============================================================
# GENERATE VALIDATION REPORT
# ============================================================
print("\n" + "=" * 80)
print("GENERATING VALIDATION REPORT")
print("=" * 80)

validation_results["v1"]["sample_prediction"] = sample_results["v1"]
validation_results["v2"]["sample_prediction"] = sample_results["v2"]
validation_results["summary"]["v1_status"] = validation_results["v1"]["status"]
validation_results["summary"]["v2_status"] = validation_results["v2"]["status"]
validation_results["summary"]["overall"] = "PASS" if (v1_all_pass and v2_all_pass) else "FAIL"

# Build markdown report
md = []
md.append("# Model Validation Report\n")
md.append(f"**Date**: {validation_results['validation_date']}\n")
md.append(f"**Project Root**: {PROJECT_ROOT}\n")
md.append("\n---\n")

# Overall summary
md.append("## Overall Summary\n")
md.append(f"| Component | Status |\n")
md.append(f"|-----------|--------|\n")
md.append(f"| V1 Pipeline | {'✅ PASS' if v1_all_pass else '❌ FAIL'} |\n")
md.append(f"| V2 Pipeline | {'✅ PASS' if v2_all_pass else '❌ FAIL'} |\n")
md.append(f"| **Overall** | **{'✅ PASS' if (v1_all_pass and v2_all_pass) else '❌ FAIL'}** |\n")

md.append("\n---\n")
md.append("## V1 Artifact Validation\n")
md.append(f"**Status**: {'✅ PASS' if v1_all_pass else '❌ FAIL'}\n\n")
md.append("| Artifact | Size (KB) | Status | Details |\n")
md.append("|----------|-----------|--------|--------|\n")
for c in validation_results["v1"]["checks"]:
    details = c.get("type", "") or c.get("status", "")
    if "vocabulary_size" in c:
        details += f" | vocab={c['vocabulary_size']}"
    if "mean_shape" in c:
        details += f" | dim={c['mean_shape']}"
    if "coef_shape" in c:
        details += f" | coef={c['coef_shape']}"
    status_icon = "✅" if c["status"].startswith("PASS") else "❌"
    md.append(f"| {c['artifact']} | {c['size_kb']:.1f} | {status_icon} | {details} |\n")

if v1_all_pass:
    meta = v1_objects.get("metadata", {})
    md.append("\n### V1 Feature Dimensions\n\n")
    fd = meta.get("feature_dimensions", {})
    for k, v in fd.items():
        md.append(f"- **{k}**: {v}\n")
    
    md.append("\n### V1 Final Metrics (System D: Hybrid)\n\n")
    metrics = meta.get("metrics", {}).get("system_D_Hybrid", {})
    if metrics:
        md.append("| Metric | Value |\n")
        md.append("|--------|-------|\n")
        for k, v in metrics.items():
            md.append(f"| {k} | {v} |\n")

md.append("\n### V1 Sample Prediction\n\n")
sp = sample_results.get("v1", {})
if sp.get("status") == "PASS":
    md.append(f"- Samples tested: {sp.get('n_samples', 0)}\n")
    for pred in sp.get("predictions", []):
        label = "Rumor" if pred["prediction"] == 1 else "Non-Rumor"
        md.append(f"- Sample {pred['index']+1}: **{label}** (conf={pred['confidence']:.4f})\n")
else:
    md.append(f"- Status: {sp.get('status', 'NOT RUN')}\n")

md.append("\n---\n")
md.append("## V2 Artifact Validation\n")
md.append(f"**Status**: {'✅ PASS' if v2_all_pass else '❌ FAIL'}\n\n")
md.append("| Artifact | Size (KB) | Status | Details |\n")
md.append("|----------|-----------|--------|--------|\n")
for c in validation_results["v2"]["checks"]:
    details = c.get("type", "") or c.get("status", "")
    if "mean_shape" in c:
        details += f" | dim={c['mean_shape']}"
    if "coef_shape" in c:
        details += f" | coef={c['coef_shape']}"
    status_icon = "✅" if c["status"].startswith("PASS") else "❌"
    md.append(f"| {c['artifact']} | {c['size_kb']:.1f} | {status_icon} | {details} |\n")

if v2_all_pass:
    meta = v2_objects.get("metadata", {})
    md.append("\n### V2 Feature Dimensions\n\n")
    fd = meta.get("feature_dimensions", {})
    for k, v in fd.items():
        md.append(f"- **{k}**: {v}\n")
    
    md.append("\n### V2 Final Metrics\n\n")
    metrics = meta.get("metrics", {})
    if metrics:
        md.append("| Metric | Value |\n")
        md.append("|--------|-------|\n")
        for k, v in metrics.items():
            md.append(f"| {k} | {v} |\n")

md.append("\n### V2 Sample Prediction\n\n")
sp = sample_results.get("v2", {})
if sp.get("status") == "PASS":
    md.append(f"- Thread ID: {sp.get('thread_id', 'N/A')}\n")
    md.append(f"- Prediction: **{sp.get('label', 'N/A')}**\n")
    md.append(f"- Confidence: {sp.get('confidence', 0):.4f}\n")
else:
    md.append(f"- Status: {sp.get('status', 'NOT RUN')}\n")

validation_results_md_path = os.path.join(REPORT_DIR, "model_validation_report.md")
with open(validation_results_md_path, 'w', encoding='utf-8') as f:
    f.writelines(md)

# Also save JSON version
json_path = os.path.join(REPORT_DIR, "model_validation_report.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Validation report saved to:")
print(f"   {validation_results_md_path}")
print(f"   {json_path}")

print("\n" + "=" * 80)
print(f"VALIDATION COMPLETE: {'✅ PASS' if (v1_all_pass and v2_all_pass) else '❌ FAIL'}")
print("=" * 80)