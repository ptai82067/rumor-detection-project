"""
Model Manager — PHEME Rumor Detection UI
=========================================
Loads serialized V1/V2 models and scalers,
performs prediction for demo use.
"""
import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import time
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer

V1_DIR = os.path.join(PROJECT_ROOT, "models", "v1")
V2_DIR = os.path.join(PROJECT_ROOT, "models", "v2")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


@st.cache_resource
def load_v1_model():
    """Load V1 artifacts: model, tfidf, scalers."""
    model_path = os.path.join(V1_DIR, "model.joblib")
    tfidf_path = os.path.join(V1_DIR, "tfidf_vectorizer.joblib")
    scaler_g_path = os.path.join(V1_DIR, "scaler_graph.joblib")

    if not all(os.path.exists(p) for p in [model_path, tfidf_path, scaler_g_path]):
        return None, None, None

    model = joblib.load(model_path)
    tfidf_vec = joblib.load(tfidf_path)
    scaler_graph = joblib.load(scaler_g_path)
    return model, tfidf_vec, scaler_graph


@st.cache_resource
def load_v2_model():
    """Load V2 artifacts: model, scaler."""
    model_path = os.path.join(V2_DIR, "model.joblib")
    scaler_path = os.path.join(V2_DIR, "scaler.joblib")

    if not all(os.path.exists(p) for p in [model_path, scaler_path]):
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


@st.cache_resource
def load_bert_model():
    """Load SentenceTransformer for MiniLM embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def predict_v2(thread_id, graph_features, prop_features, source_text):
    """
    V2 prediction pipeline: MiniLM(384) + Prop(4) + Graph(14) -> 402-dim.
    Returns (prediction, confidence, inference_time_ms, n_features)
    """
    start = time.time()
    
    model, scaler = load_v2_model()
    if model is None:
        return None, None, 0, 0
    
    bert = load_bert_model()
    
    # Step 1: MiniLM embedding
    emb = bert.encode([source_text])[0]  # 384-dim
    
    # Step 2: Propagation features (4)
    prop = np.array([[
        prop_features['thread_size'],
        prop_features['max_depth'],
        prop_features['avg_depth'],
        prop_features['reply_rate']
    ]])
    
    # Step 3: Graph features (14)
    graph_cols = [
        'thread_depth', 'num_nodes', 'num_edges', 'avg_branching_factor',
        'max_branching_factor', 'source_reply_count', 'leaf_ratio', 'avg_depth',
        'source_pagerank', 'avg_pagerank', 'source_centrality', 'avg_centrality',
        'user_rumor_ratio', 'unique_users'
    ]
    graph = np.array([[graph_features[c] for c in graph_cols]])
    
    # Step 4: Concatenate
    X = np.hstack([emb.reshape(1, -1), prop, graph])  # (1, 402)
    
    # Step 5: Scale
    X_scaled = scaler.transform(X)
    
    # Step 6: Predict
    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = float(max(proba))
    
    elapsed = (time.time() - start) * 1000
    
    return pred, confidence, round(elapsed, 2), 402


def predict_v1(post_id, text):
    """
    V1 prediction pipeline: TF-IDF(5000) + MiniLM(384) + Graph(14) -> 5398-dim.
    Returns (prediction, confidence, inference_time_ms, n_features)
    """
    start = time.time()
    
    model, tfidf_vec, scaler_graph = load_v1_model()
    if model is None:
        return None, None, 0, 0
    
    bert = load_bert_model()
    
    # Step 1: TF-IDF
    X_tfidf = tfidf_vec.transform([text]).toarray()  # (1, 5000)
    
    # Step 2: MiniLM
    emb = bert.encode([text])  # (1, 384)
    
    # Step 3: Graph features - look up from pheme_features_with_graph.csv
    df_graph = pd.read_csv(os.path.join(DATA_DIR, "pheme_features_with_graph.csv"))
    g_row = df_graph[df_graph['post_id'] == post_id]
    
    if len(g_row) == 0:
        # Fallback: use zero features
        graph = np.zeros((1, 14))
    else:
        from utils.graph_features import get_graph_feature_columns
        gcols = get_graph_feature_columns()
        graph = scaler_graph.transform(g_row[gcols].values)  # (1, 14)
    
    # Step 4: Concatenate [TF-IDF(5000), MiniLM(384), Graph(14)] = 5398
    X = np.hstack([X_tfidf, emb, graph])  # (1, 5398)
    
    # Step 5: Predict
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(max(proba))
    
    elapsed = (time.time() - start) * 1000
    
    return pred, confidence, round(elapsed, 2), 5398