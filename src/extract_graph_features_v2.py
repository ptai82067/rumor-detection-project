"""
Graph Feature Extraction from Knowledge Graph v2 — PHEME Rumor Detection

This script extracts 14 thread-level graph features from the PHEME KG v2
(or equivalently from pheme_features_with_graph.csv which contains the same data).

Output: data/processed/graph_features_v2.csv
  - Column 1: thread_id
  - Columns 2-15: 14 graph features (thread-level aggregated)
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_WITH_GRAPH = os.path.join(BASE_DIR, "data", "processed", "pheme_features_with_graph.csv")
FEATURES_BASIC = os.path.join(BASE_DIR, "data", "processed", "pheme_features.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "graph_features_v2.csv")


def build_thread_reply_trees(df: pd.DataFrame) -> Dict[int, Tuple[nx.DiGraph, Dict]]:
    """
    Build per-thread reply trees from the feature dataframe.
    
    Returns dict: thread_id -> (NetworkX DiGraph, metadata dict)
    """
    logger.info("Building per-thread reply trees from feature data...")
    
    thread_trees = {}
    
    for thread_id, group in df.groupby('thread_id'):
        G = nx.DiGraph()
        meta = {
            'source_post_id': None,
            'user_ids': set(),
            'user_rumor_ratios': [],
        }
        
        tid = int(thread_id)
        
        for _, row in group.iterrows():
            post_id = int(row['post_id'])
            user_id = int(row['user_id'])
            depth = int(row['depth'])
            reply_to_raw = row.get('reply_to', None)
            is_source = bool(row.get('is_source', False))
            
            # Add node
            G.add_node(post_id, depth=depth, user_id=user_id, is_source=is_source or depth == 0)
            
            # Track metadata
            meta['user_ids'].add(user_id)
            
            # Get user prior rumor ratio if available
            if 'user_prior_rumor_ratio' in df.columns:
                val = row['user_prior_rumor_ratio']
                if pd.notna(val):
                    meta['user_rumor_ratios'].append(float(val))
            
            # Set source post
            if depth == 0 or is_source:
                if meta['source_post_id'] is None:
                    meta['source_post_id'] = post_id
            
            # Add edge from parent (reply_to) to this post
            if pd.notna(reply_to_raw) and str(reply_to_raw) != 'nan' and str(reply_to_raw).strip():
                try:
                    parent_id = int(float(str(reply_to_raw)))
                    if parent_id in group['post_id'].values:
                        G.add_edge(parent_id, post_id, relation='repliesTo')
                except (ValueError, TypeError):
                    pass
        
        # Store metadata
        meta['num_users'] = len(meta['user_ids'])
        meta['avg_user_rumor_ratio'] = np.mean(meta['user_rumor_ratios']) if meta['user_rumor_ratios'] else 0.0
        
        thread_trees[tid] = (G, meta)
    
    return thread_trees


def compute_thread_depth(G: nx.DiGraph, meta: Dict) -> int:
    """Maximum depth of the reply tree."""
    depths = nx.get_node_attributes(G, 'depth')
    return max(depths.values()) if depths else 0


def compute_num_nodes(G: nx.DiGraph, meta: Dict) -> int:
    """Total number of nodes in the thread."""
    return G.number_of_nodes()


def compute_num_edges(G: nx.DiGraph, meta: Dict) -> int:
    """Total number of edges in the reply tree."""
    return G.number_of_edges()


def compute_avg_branching_factor(G: nx.DiGraph, meta: Dict) -> float:
    """Average number of replies per node."""
    if G.number_of_nodes() == 0:
        return 0.0
    out_degrees = [d for _, d in G.out_degree()]
    return float(np.mean(out_degrees)) if out_degrees else 0.0


def compute_max_branching_factor(G: nx.DiGraph, meta: Dict) -> int:
    """Maximum number of replies for any single node."""
    if G.number_of_nodes() == 0:
        return 0
    out_degrees = [d for _, d in G.out_degree()]
    return max(out_degrees) if out_degrees else 0


def compute_source_reply_count(G: nx.DiGraph, meta: Dict) -> int:
    """Number of direct replies to the source post."""
    source_id = meta.get('source_post_id')
    if source_id is None or source_id not in G:
        return 0
    return G.out_degree(source_id)


def compute_leaf_ratio(G: nx.DiGraph, meta: Dict) -> float:
    """Ratio of leaf nodes (nodes with no outgoing edges / no replies)."""
    if G.number_of_nodes() == 0:
        return 0.0
    leaf_count = sum(1 for node in G.nodes() if G.out_degree(node) == 0)
    return leaf_count / G.number_of_nodes()


def compute_avg_depth(G: nx.DiGraph, meta: Dict) -> float:
    """Average depth of all nodes in the thread."""
    depths = nx.get_node_attributes(G, 'depth')
    if not depths:
        return 0.0
    return float(np.mean(list(depths.values())))


def compute_pagerank(G: nx.DiGraph, meta: Dict) -> Tuple[float, float]:
    """
    Compute PageRank of source node and average PageRank across thread.
    Returns: (source_pagerank, avg_pagerank)
    """
    if G.number_of_nodes() == 0:
        return (0.0, 0.0)
    
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-4)
    
    avg_pr = float(np.mean(list(pr.values()))) if pr else 0.0
    
    source_id = meta.get('source_post_id')
    source_pr = pr.get(source_id, 0.0) if source_id else 0.0
    
    return (source_pr, avg_pr)


def compute_centrality(G: nx.DiGraph, meta: Dict) -> Tuple[float, float]:
    """
    Compute degree centrality of source node and average degree centrality.
    Returns: (source_centrality, avg_centrality)
    """
    if G.number_of_nodes() == 0:
        return (0.0, 0.0)
    
    dc = nx.degree_centrality(G)
    
    avg_dc = float(np.mean(list(dc.values()))) if dc else 0.0
    
    source_id = meta.get('source_post_id')
    source_dc = dc.get(source_id, 0.0) if source_id else 0.0
    
    return (source_dc, avg_dc)


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 14 thread-level graph features and return as DataFrame.
    """
    logger.info("Building reply trees...")
    thread_trees = build_thread_reply_trees(df)
    logger.info(f"Built {len(thread_trees)} thread reply trees")
    
    records = []
    
    for tid, (G, meta) in thread_trees.items():
        thread_depth = compute_thread_depth(G, meta)
        num_nodes = compute_num_nodes(G, meta)
        num_edges = compute_num_edges(G, meta)
        avg_branching = compute_avg_branching_factor(G, meta)
        max_branching = compute_max_branching_factor(G, meta)
        source_reply_cnt = compute_source_reply_count(G, meta)
        leaf_ratio = compute_leaf_ratio(G, meta)
        avg_depth = compute_avg_depth(G, meta)
        source_pr, avg_pr = compute_pagerank(G, meta)
        source_dc, avg_dc = compute_centrality(G, meta)
        
        records.append({
            'thread_id': tid,
            'thread_depth': thread_depth,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_branching_factor': round(avg_branching, 6),
            'max_branching_factor': max_branching,
            'source_reply_count': source_reply_cnt,
            'leaf_ratio': round(leaf_ratio, 6),
            'avg_depth': round(avg_depth, 6),
            'source_pagerank': round(source_pr, 8),
            'avg_pagerank': round(avg_pr, 8),
            'source_centrality': round(source_dc, 8),
            'avg_centrality': round(avg_dc, 8),
            'user_rumor_ratio': round(meta['avg_user_rumor_ratio'], 6),
            'unique_users': meta['num_users'],
        })
    
    result_df = pd.DataFrame(records)
    return result_df


def validate_output(df: pd.DataFrame):
    """Validate the output dataframe."""
    print(f"\n{'=' * 60}")
    print(f"VALIDATION - graph_features_v2.csv")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"Expected threads: 5802")
    print(f"Actual threads:   {len(df)}")
    
    if len(df) != 5802:
        logger.warning(f"Expected 5802 threads but got {len(df)}")
    
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"Null count: {total_nulls}")
    if total_nulls > 0:
        print(f"Nulls per column:")
        print(null_counts[null_counts > 0].to_string())
    
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    
    print(f"\nFeature statistics:")
    print(df.describe().round(4).to_string())
    
    ok = len(df) == 5802 and total_nulls == 0
    print(f"\n[OK] Output shape correct: {len(df)} == 5802: {len(df) == 5802}")
    print(f"[OK] No null values: {total_nulls == 0}")
    return ok


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("EXTRACTING GRAPH FEATURES V2 FROM PHEME DATA")
    logger.info("=" * 60)
    
    # Try to load data with graph features first (has user_prior_rumor_ratio etc.)
    if os.path.exists(FEATURES_WITH_GRAPH):
        df = pd.read_csv(FEATURES_WITH_GRAPH, dtype={'reply_to': str})
        logger.info(f"Loaded {FEATURES_WITH_GRAPH}: {df.shape}")
    else:
        df = pd.read_csv(FEATURES_BASIC, dtype={'reply_to': str})
        logger.info(f"Loaded {FEATURES_BASIC}: {df.shape} (no user_prior_rumor_ratio)")
    
    # Clean reply_to
    df['reply_to'] = df['reply_to'].replace('nan', pd.NA)
    
    # Compute features
    result = compute_all_features(df)
    
    # Validate
    if not validate_output(result):
        logger.warning("Validation found issues, but saving anyway")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"[OK] Graph features v2 saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()