"""
Data Loader — PHEME Rumor Detection UI
=======================================
Shared data loading with Streamlit caching.
All functions use @st.cache_data to load data only once.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
ABLATION_DIR = os.path.join(RESULT_DIR, "ablation")


@st.cache_data
def load_pheme_features():
    """Load base PHEME features dataset."""
    path = os.path.join(DATA_DIR, "pheme_features.csv")
    df = pd.read_csv(path, dtype={'reply_to': str})
    return df


@st.cache_data
def load_pheme_features_with_graph():
    """Load PHEME features with graph features (for V1)."""
    path = os.path.join(DATA_DIR, "pheme_features_with_graph.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None


@st.cache_data
def load_graph_features_v2():
    """Load thread-level graph features (for V2)."""
    path = os.path.join(DATA_DIR, "graph_features_v2.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_dataset_statistics():
    """Load dataset statistics."""
    path = os.path.join(RESULT_DIR, "pheme_dataset_statistics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        stats = dict(zip(df['Metric'], df['Value']))
        return stats
    return {}


@st.cache_data
def load_final_metrics():
    """Load final metrics table."""
    path = os.path.join(PROJECT_ROOT, "final_metrics_table.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None


@st.cache_data
def load_ablation_table():
    """Load ablation study table."""
    path = os.path.join(ABLATION_DIR, "ablation_table.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None


@st.cache_data
def load_v1_metadata():
    """Load V1 model metadata."""
    path = os.path.join(MODELS_DIR, "v1", "metadata.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_v2_metadata():
    """Load V2 model metadata."""
    path = os.path.join(MODELS_DIR, "v2", "metadata.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_ontology_text(version=1):
    """Load ontology TTL file and return first N lines."""
    path = os.path.join(PROJECT_ROOT, "ontology", f"pheme_ontology_v{version}.ttl")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


@st.cache_data
def get_sample_threads():
    """Get list of sample thread IDs with labels and source text for UI selection."""
    base_df = load_pheme_features()
    graph_df = load_graph_features_v2()

    samples = []
    for tid in graph_df['thread_id'].head(50).tolist():
        group = base_df[base_df['thread_id'] == tid]
        if len(group) > 0:
            source = group[group['depth'] == 0].iloc[0] if len(group[group['depth'] == 0]) > 0 else group.iloc[0]
            label = "Rumor" if int(source['label']) == 1 else "Non-Rumor"
            text = str(source['text'])[:80]
            
            # Get thread-level features from graph_df
            g_row = graph_df[graph_df['thread_id'] == tid]
            num_nodes = int(g_row['num_nodes'].iloc[0]) if len(g_row) > 0 else 0
            
            samples.append({
                'thread_id': int(tid),
                'label': label,
                'source_text': text,
                'num_posts': num_nodes
            })
    
    return samples


@st.cache_data
def get_thread_data(thread_id):
    """Get full post-level data for a specific thread."""
    base_df = load_pheme_features()
    thread_posts = base_df[base_df['thread_id'] == thread_id].copy()
    
    if len(thread_posts) == 0:
        return None
    
    thread_posts['is_source'] = thread_posts['depth'] == 0
    return thread_posts


@st.cache_data
def get_v2_features_for_thread(thread_id):
    """Get V2 features (graph + propagation) for a specific thread."""
    graph_df = load_graph_features_v2()
    base_df = load_pheme_features()
    
    g_row = graph_df[graph_df['thread_id'] == thread_id]
    if len(g_row) == 0:
        return None
    
    graph_cols = [
        'thread_depth', 'num_nodes', 'num_edges', 'avg_branching_factor',
        'max_branching_factor', 'source_reply_count', 'leaf_ratio', 'avg_depth',
        'source_pagerank', 'avg_pagerank', 'source_centrality', 'avg_centrality',
        'user_rumor_ratio', 'unique_users'
    ]
    graph_features = {col: float(g_row[col].iloc[0]) for col in graph_cols}
    
    # Propagation features
    thread_posts = base_df[base_df['thread_id'] == thread_id]
    thread_size = int(thread_posts['thread_size'].iloc[0]) if 'thread_size' in thread_posts.columns else len(thread_posts)
    max_depth = int(thread_posts['max_depth'].iloc[0]) if 'max_depth' in thread_posts.columns else int(thread_posts['depth'].max())
    avg_depth = float(thread_posts['depth'].mean())
    reply_rate = len(thread_posts) / (thread_size + 1) if thread_size > 0 else 0
    
    prop_features = {
        'thread_size': thread_size,
        'max_depth': max_depth,
        'avg_depth': round(avg_depth, 4),
        'reply_rate': round(reply_rate, 4)
    }
    
    return graph_features, prop_features