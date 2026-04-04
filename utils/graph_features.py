"""
Graph Feature Extraction for PHEME Rumor Detection

This module extracts features from the PHEME knowledge graph to enhance
the baseline rumor detection model. Features include:
- Node centrality (in_degree, out_degree, pagerank)
- User reputation features
- Thread structure features

Input: data/processed/pheme_kg.ttl (Knowledge Graph)
       data/processed/pheme_features.csv (Base features)
Output: DataFrame with post_id and graph-derived features
"""

import pandas as pd
import numpy as np
import networkx as nx
from rdflib import Graph, Namespace, RDF
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define namespace
EX = Namespace("http://example.org/pheme#")


def extract_post_id_from_uri(uri) -> Optional[int]:
    """Extract post ID from RDF URI."""
    uri_str = str(uri)
    if '/post/' in uri_str:
        try:
            post_part = uri_str.split('/post/')[-1].split('#')[0].split('?')[0]
            return int(post_part)
        except ValueError:
            logger.warning(f"Could not extract post ID from URI: {uri_str}")
            return None
    return None


def build_reply_graph(kg_path: str = "data/processed/pheme_kg.ttl") -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from the knowledge graph's repliesTo relationships.
    
    The graph direction is: parent -> child (source post -> reply)
    This allows us to compute features like:
    - in_degree: number of direct replies to a post
    - out_degree: whether a post is a reply (0 or 1)
    - pagerank: authority score in the conversation network
    """
    logger.info(f"Building reply graph from {kg_path}")
    
    # Load knowledge graph
    g = Graph()
    g.parse(kg_path, format="turtle")
    logger.info(f"Loaded KG with {len(g)} triples")
    
    # Build reply graph
    G = nx.DiGraph()
    
    # Count edges for logging
    edge_count = 0
    for subject, predicate, obj in g.triples((None, EX.repliesTo, None)):
        child_id = extract_post_id_from_uri(subject)
        parent_id = extract_post_id_from_uri(obj)
        
        if child_id is not None and parent_id is not None:
            # Direction: parent -> child (source -> reply)
            G.add_edge(parent_id, child_id)
            edge_count += 1
    
    logger.info(f"Built reply graph with {G.number_of_nodes()} nodes and {edge_count} edges")
    return G


def compute_node_centrality_features(G: nx.DiGraph, all_post_ids: set) -> pd.DataFrame:
    """
    Compute node centrality features for all posts.
    
    Features:
    - in_degree: Number of direct replies received
    - out_degree: Number of replies sent (0 or 1 in tree structure)
    - pagerank: PageRank score indicating authority in conversation
    - betweenness_centrality: How often a post acts as a bridge
    - closeness_centrality: How close a post is to all other posts
    """
    logger.info("Computing node centrality features...")
    
    # Compute PageRank
    logger.info("Computing PageRank...")
    pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100)
    
    # Compute betweenness centrality (sample for large graphs)
    logger.info("Computing betweenness centrality...")
    if G.number_of_nodes() > 10000:
        # Sample k nodes for faster computation
        betweenness = nx.betweenness_centrality(G, k=1000, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    
    # Compute closeness centrality (on undirected version for better connectivity)
    logger.info("Computing closeness centrality...")
    G_undirected = G.to_undirected()
    closeness = nx.closeness_centrality(G_undirected)
    
    # Build feature dataframe
    features = []
    for post_id in all_post_ids:
        post_features = {
            'post_id': post_id,
            'node_in_degree': G.in_degree(post_id) if post_id in G else 0,
            'node_out_degree': G.out_degree(post_id) if post_id in G else 0,
            'pagerank_score': pagerank_scores.get(post_id, 0.0),
            'betweenness_centrality': betweenness.get(post_id, 0.0),
            'closeness_centrality': closeness.get(post_id, 0.0),
        }
        features.append(post_features)
    
    df_features = pd.DataFrame(features)
    logger.info(f"Computed centrality features for {len(df_features)} posts")
    return df_features


def compute_user_reputation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user reputation features based on historical behavior.
    
    Features:
    - user_prior_rumor_ratio: Historical rate of posting rumors
    - user_post_count: Total number of posts by user
    - user_thread_count: Number of unique threads user participates in
    - user_avg_depth: Average depth of user's posts
    """
    logger.info("Computing user reputation features...")
    
    # Group by user_id
    user_stats = df.groupby('user_id').agg(
        user_post_count=('post_id', 'count'),
        user_thread_count=('thread_id', 'nunique'),
        user_avg_depth=('depth', 'mean'),
        user_rumor_count=('label', 'sum'),
    ).reset_index()
    
    # Compute rumor ratio
    user_stats['user_prior_rumor_ratio'] = user_stats['user_rumor_count'] / user_stats['user_post_count']
    
    # Drop intermediate columns
    user_stats = user_stats.drop(columns=['user_rumor_count'])
    
    logger.info(f"Computed reputation features for {len(user_stats)} users")
    return user_stats


def compute_thread_structure_features(G: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute thread structure features.
    
    Features:
    - subtree_reply_count: Total replies in the subtree rooted at this post
    - sibling_count: Number of posts replying to the same parent
    - position_in_thread: Order of post in thread (by time)
    """
    logger.info("Computing thread structure features...")
    
    # Build children mapping from reply graph
    children_map = defaultdict(list)
    for parent, child in G.edges():
        children_map[parent].append(child)
    
    # Compute subtree sizes using BFS
    def compute_subtree_size(node: int, children_map: dict, memo: dict) -> int:
        """Recursively compute subtree size with memoization."""
        if node in memo:
            return memo[node]
        
        size = 0
        for child in children_map.get(node, []):
            size += 1 + compute_subtree_size(child, children_map, memo)
        
        memo[node] = size
        return size
    
    memo = {}
    subtree_sizes = {}
    for node in G.nodes():
        subtree_sizes[node] = compute_subtree_size(node, children_map, memo)
    
    # Compute sibling counts
    # First, build parent mapping (child -> parent)
    parent_map = {}
    for parent, child in G.edges():
        parent_map[child] = parent
    
    # Count siblings (posts with same parent)
    sibling_counts = defaultdict(int)
    for child, parent in parent_map.items():
        sibling_counts[child] = len([c for c, p in parent_map.items() if p == parent]) - 1
    
    # Compute position in thread (based on time)
    df_sorted = df.sort_values(['thread_id', 'time'])
    position_in_thread = df_sorted.groupby('thread_id').cumcount() + 1
    position_map = dict(zip(df_sorted['post_id'], position_in_thread))
    
    # Build feature dataframe
    features = []
    for _, row in df.iterrows():
        post_id = row['post_id']
        features.append({
            'post_id': post_id,
            'subtree_reply_count': subtree_sizes.get(post_id, 0),
            'sibling_count': sibling_counts.get(post_id, 0),
            'position_in_thread': position_map.get(post_id, 0),
        })
    
    df_features = pd.DataFrame(features)
    logger.info(f"Computed thread structure features for {len(df_features)} posts")
    return df_features


def compute_source_authority_features(df: pd.DataFrame, user_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute source authority features for thread originators.
    
    Features:
    - source_user_credibility: Credibility score of thread source user
    - source_network_size: Network reach of source user (degree in reply graph)
    """
    logger.info("Computing source authority features...")
    
    # Identify source posts (depth = 0)
    source_posts = df[df['depth'] == 0][['thread_id', 'user_id', 'post_id']].copy()
    
    # Get user credibility
    source_posts = source_posts.merge(
        user_features[['user_id', 'user_prior_rumor_ratio', 'user_post_count']],
        on='user_id',
        how='left'
    )
    
    # Source credibility is inverse of rumor ratio (non-rumor posting history = more credible)
    source_posts['source_user_credibility'] = 1 - source_posts['user_prior_rumor_ratio']
    source_posts['source_network_size'] = source_posts['user_post_count']
    
    # Map back to all posts in thread
    thread_source_features = source_posts[['thread_id', 'source_user_credibility', 'source_network_size']]
    
    df_features = df[['post_id', 'thread_id']].merge(
        thread_source_features,
        on='thread_id',
        how='left'
    )
    
    # Fill NaN for any posts not in a thread (shouldn't happen, but safety)
    df_features['source_user_credibility'] = df_features['source_user_credibility'].fillna(0.5)
    df_features['source_network_size'] = df_features['source_network_size'].fillna(0)
    
    logger.info(f"Computed source authority features for {len(df_features)} posts")
    return df_features[['post_id', 'source_user_credibility', 'source_network_size']]


def extract_all_graph_features(
    df: pd.DataFrame,
    kg_path: str = "data/processed/pheme_kg.ttl"
) -> pd.DataFrame:
    """
    Extract all graph-derived features and merge with the original dataframe.
    
    This is the main entry point for graph feature extraction.
    
    Args:
        df: Original feature dataframe with post_id column
        kg_path: Path to the knowledge graph TTL file
    
    Returns:
        DataFrame with original columns plus all graph features
    """
    logger.info("Starting complete graph feature extraction pipeline...")
    
    # Get all post IDs
    all_post_ids = set(df['post_id'].unique())
    logger.info(f"Processing {len(all_post_ids)} unique posts")
    
    # Step 1: Build reply graph from KG
    G = build_reply_graph(kg_path)
    
    # Step 2: Compute node centrality features
    centrality_features = compute_node_centrality_features(G, all_post_ids)
    
    # Step 3: Compute user reputation features
    user_features = compute_user_reputation_features(df)
    
    # Step 4: Compute thread structure features
    thread_features = compute_thread_structure_features(G, df)
    
    # Step 5: Compute source authority features
    source_features = compute_source_authority_features(df, user_features)
    
    # Step 6: Merge all features
    logger.info("Merging all graph features...")
    
    # Start with original dataframe
    df_enriched = df.copy()
    
    # Merge centrality features
    df_enriched = df_enriched.merge(centrality_features, on='post_id', how='left')
    
    # Merge user features
    df_enriched = df_enriched.merge(user_features, on='user_id', how='left')
    
    # Merge thread structure features
    df_enriched = df_enriched.merge(thread_features, on='post_id', how='left')
    
    # Merge source authority features
    df_enriched = df_enriched.merge(source_features, on='post_id', how='left')
    
    # Fill any remaining NaN values with 0
    graph_feature_cols = [
        'node_in_degree', 'node_out_degree', 'pagerank_score',
        'betweenness_centrality', 'closeness_centrality',
        'user_prior_rumor_ratio', 'user_post_count', 'user_thread_count', 'user_avg_depth',
        'subtree_reply_count', 'sibling_count', 'position_in_thread',
        'source_user_credibility', 'source_network_size'
    ]
    
    for col in graph_feature_cols:
        if col in df_enriched.columns:
            df_enriched[col] = df_enriched[col].fillna(0)
    
    logger.info(f"Graph feature extraction complete. Added {len(graph_feature_cols)} new features.")
    logger.info(f"Final dataframe shape: {df_enriched.shape}")
    
    return df_enriched


def get_graph_feature_columns() -> List[str]:
    """Return the list of graph-derived feature column names."""
    return [
        # Node centrality
        'node_in_degree',
        'node_out_degree', 
        'pagerank_score',
        'betweenness_centrality',
        'closeness_centrality',
        # User reputation
        'user_prior_rumor_ratio',
        'user_post_count',
        'user_thread_count',
        'user_avg_depth',
        # Thread structure
        'subtree_reply_count',
        'sibling_count',
        'position_in_thread',
        # Source authority
        'source_user_credibility',
        'source_network_size',
    ]


def get_feature_sets() -> Dict[str, List[str]]:
    """
    Return predefined feature sets for ablation study.
    
    Returns:
        Dictionary mapping feature set name to list of column names
    """
    return {
        'node_centrality': [
            'node_in_degree',
            'node_out_degree',
            'pagerank_score',
        ],
        'user_reputation': [
            'user_prior_rumor_ratio',
            'user_post_count',
        ],
        'thread_structure': [
            'subtree_reply_count',
            'sibling_count',
            'position_in_thread',
        ],
        'source_authority': [
            'source_user_credibility',
            'source_network_size',
        ],
        'all_centrality': [
            'node_in_degree',
            'node_out_degree',
            'pagerank_score',
            'betweenness_centrality',
            'closeness_centrality',
        ],
        'all_graph_features': get_graph_feature_columns(),
    }


if __name__ == "__main__":
    # Example usage: Extract graph features and save to CSV
    print("Loading base features...")
    df = pd.read_csv("data/processed/pheme_features.csv")
    print(f"Loaded {len(df)} posts")
    
    print("\nExtracting graph features...")
    df_enriched = extract_all_graph_features(df)
    
    print(f"\nEnriched dataframe shape: {df_enriched.shape}")
    print(f"Original columns: {len(df.columns)}")
    print(f"New columns: {len(df_enriched.columns) - len(df.columns)}")
    print(f"Graph feature columns: {get_graph_feature_columns()}")
    
    # Save enriched features (optional - for inspection only)
    # df_enriched.to_csv("data/processed/pheme_features_with_graph.csv", index=False)
    # print("\nEnriched features saved to data/processed/pheme_features_with_graph.csv")