"""
BERT Feature Fusion Module for PHEME Rumor Detection

This module integrates sentence-level semantic embeddings from pre-trained 
MiniLM encoder with handcrafted graph features for enhanced rumor detection.

Key Features:
- Extract 384-dimensional sentence embeddings using sentence-transformers
- Normalize and fuse embeddings with graph handcrafted features
- Maintain strict train/test separation to prevent data leakage
- Compatible with existing pipeline (RANDOM_STATE=42, stratified split)

Usage:
    from utils.bert_feature_fusion import (
        extract_sentence_embeddings,
        fuse_bert_with_features,
        create_fusion_feature_matrix
    )

Note: This module NEVER modifies frozen data files.
      All operations are performed in-memory.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BERT_MODEL_NAME = "all-MiniLM-L6-v2"
BERT_EMBEDDING_DIM = 384


def load_bert_model(model_name: str = BERT_MODEL_NAME) -> SentenceTransformer:
    """
    Load pre-trained SentenceTransformer model.
    
    Args:
        model_name: Name of the pre-trained model
        
    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading BERT model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded successfully. Embedding dimension: {BERT_EMBEDDING_DIM}")
    return model


def extract_sentence_embeddings(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract sentence-level semantic embeddings from text using pre-trained MiniLM encoder.
    
    This function produces pooled sentence embeddings (384 dimensions) rather than
    raw [CLS] token hidden states. The pooling operation combines token embeddings
    to create a fixed-size representation of the entire sentence meaning.
    
    Args:
        texts: List of text strings to encode
        model: Pre-loaded SentenceTransformer model (if None, loads default model)
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
        
    Returns:
        numpy array of shape (n_texts, 384) containing sentence embeddings
    """
    if model is None:
        model = load_bert_model()
    
    logger.info(f"Extracting sentence embeddings for {len(texts)} texts")
    
    # Handle empty or NaN texts
    clean_texts = []
    for text in texts:
        if pd.isna(text) or str(text).strip() == "":
            clean_texts.append("")  # Empty string will produce zero-like embedding
        else:
            clean_texts.append(str(text))
    
    # Extract embeddings
    embeddings = model.encode(
        clean_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Critical for preventing data leakage:
    - Fit scaler ONLY on training data
    - Transform both train and test using the same scaler
    
    Args:
        X_train: Training features
        X_test: Test features
        scaler: Pre-fitted scaler (if None and fit=True, creates new scaler)
        fit: Whether to fit the scaler (False for test data)
        
    Returns:
        Tuple of (normalized_train, normalized_test, scaler)
    """
    if fit:
        if scaler is None:
            scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def fuse_bert_with_features(
    X_bert_train: np.ndarray,
    X_bert_test: np.ndarray,
    X_other_train: np.ndarray,
    X_other_test: np.ndarray,
    normalize_other: bool = True,
    other_scaler: Optional[StandardScaler] = None
) -> Tuple[Union[np.ndarray, csr_matrix], Union[np.ndarray, csr_matrix], Optional[StandardScaler]]:
    """
    Fuse BERT embeddings with other features (graph, propagation).
    
    Architecture:
        [BERT embeddings (384)] + [normalized other features] → classifier
    
    Args:
        X_bert_train: BERT embeddings for training set
        X_bert_test: BERT embeddings for test set
        X_other_train: Other features (graph, propagation) for training set
        X_other_test: Other features (graph, propagation) for test set
        normalize_other: Whether to normalize other features
        other_scaler: Pre-fitted scaler for other features
        
    Returns:
        Tuple of (fused_train, fused_test, other_scaler_if_fitted)
    """
    logger.info(f"Fusing BERT embeddings ({X_bert_train.shape[1]}-dim) with other features ({X_other_train.shape[1]}-dim)")
    
    # Normalize other features if requested
    if normalize_other:
        X_other_train_norm, X_other_test_norm, other_scaler = normalize_features(
            X_other_train, X_other_test, other_scaler, fit=(other_scaler is None)
        )
    else:
        X_other_train_norm = X_other_train
        X_other_test_norm = X_other_test
    
    # Concatenate features
    X_train_fused = np.hstack([X_bert_train, X_other_train_norm])
    X_test_fused = np.hstack([X_bert_test, X_other_test_norm])
    
    logger.info(f"Fused feature dimension: {X_train_fused.shape[1]}")
    
    return X_train_fused, X_test_fused, other_scaler


def create_fusion_feature_matrix(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_full: pd.DataFrame,  # For extracting graph features (uses full dataset context)
    text_column: str = 'text',
    graph_feature_columns: Optional[List[str]] = None,
    propagation_columns: List[str] = ['is_reply', 'thread_size', 'children_count', 'depth'],
    bert_model: Optional[SentenceTransformer] = None,
    include_propagation: bool = True,
    include_graph: bool = True
) -> Dict:
    """
    Create complete feature matrices for BERT + graph fusion experiments.
    
    This is the main entry point for creating feature matrices for the ablation study.
    
    Args:
        df_train: Training subset of the dataframe
        df_test: Test subset of the dataframe
        df_full: Full dataframe (for graph feature extraction context)
        text_column: Name of the text column
        graph_feature_columns: List of graph feature column names
        propagation_columns: List of propagation feature column names
        bert_model: Pre-loaded BERT model
        include_propagation: Whether to include propagation features
        include_graph: Whether to include graph features
        
    Returns:
        Dictionary containing:
        - X_train, X_test: Final fused feature matrices
        - X_bert_train, X_bert_test: BERT embeddings only
        - X_propagation_train, X_propagation_test: Propagation features only
        - X_graph_train, X_graph_test: Graph features only
        - propagation_scaler: Fitted scaler for propagation features
        - graph_scaler: Fitted scaler for graph features
    """
    if graph_feature_columns is None:
        from utils.graph_features import get_graph_feature_columns
        graph_feature_columns = get_graph_feature_columns()
    
    result = {}
    
    # 1. Extract BERT embeddings
    logger.info("Step 1: Extracting BERT sentence embeddings")
    X_bert_train = extract_sentence_embeddings(df_train[text_column].tolist(), bert_model)
    X_bert_test = extract_sentence_embeddings(df_test[text_column].tolist(), bert_model)
    
    result['X_bert_train'] = X_bert_train
    result['X_bert_test'] = X_bert_test
    
    # 2. Extract propagation features
    if include_propagation:
        logger.info("Step 2: Extracting propagation features")
        X_prop_train = df_train[propagation_columns].values
        X_prop_test = df_test[propagation_columns].values
        
        # Normalize propagation features
        X_prop_train_norm, X_prop_test_norm, prop_scaler = normalize_features(
            X_prop_train, X_prop_test, fit=True
        )
        
        result['X_propagation_train'] = X_prop_train_norm
        result['X_propagation_test'] = X_prop_test_norm
        result['propagation_scaler'] = prop_scaler
    else:
        result['X_propagation_train'] = None
        result['X_propagation_test'] = None
        result['propagation_scaler'] = None
    
    # 3. Extract graph features (from df_full to maintain consistency with existing pipeline)
    if include_graph:
        logger.info("Step 3: Extracting graph features")
        X_graph_train = df_full.loc[df_train.index, graph_feature_columns].values
        X_graph_test = df_full.loc[df_test.index, graph_feature_columns].values
        
        # Normalize graph features
        X_graph_train_norm, X_graph_test_norm, graph_scaler = normalize_features(
            X_graph_train, X_graph_test, fit=True
        )
        
        result['X_graph_train'] = X_graph_train_norm
        result['X_graph_test'] = X_graph_test_norm
        result['graph_scaler'] = graph_scaler
    else:
        result['X_graph_train'] = None
        result['X_graph_test'] = None
        result['graph_scaler'] = None
    
    # 4. Fuse all selected features
    logger.info("Step 4: Fusing features")
    
    # Start with BERT embeddings
    X_train_fused = X_bert_train
    X_test_fused = X_bert_test
    
    # Add propagation if included
    if include_propagation and result['X_propagation_train'] is not None:
        X_train_fused = np.hstack([X_train_fused, result['X_propagation_train']])
        X_test_fused = np.hstack([X_test_fused, result['X_propagation_test']])
    
    # Add graph if included
    if include_graph and result['X_graph_train'] is not None:
        X_train_fused = np.hstack([X_train_fused, result['X_graph_train']])
        X_test_fused = np.hstack([X_test_fused, result['X_graph_test']])
    
    result['X_train'] = X_train_fused
    result['X_test'] = X_test_fused
    
    logger.info(f"Final fused feature dimension: {X_train_fused.shape[1]}")
    
    return result


def get_feature_dimension_breakdown(
    include_propagation: bool = True,
    include_graph: bool = True
) -> Dict[str, int]:
    """
    Get the dimension breakdown for different feature combinations.
    
    Args:
        include_propagation: Whether propagation features are included
        include_graph: Whether graph features are included
        
    Returns:
        Dictionary with dimension information
    """
    breakdown = {
        'bert_embeddings': BERT_EMBEDDING_DIM,
        'propagation_features': 4 if include_propagation else 0,
        'graph_features': 14 if include_graph else 0,
    }
    breakdown['total'] = sum(breakdown.values())
    
    return breakdown


if __name__ == "__main__":
    # Example usage and testing
    print("BERT Feature Fusion Module")
    print("=" * 60)
    
    # Test model loading
    print("\n1. Testing model loading...")
    model = load_bert_model()
    print(f"   Model loaded: {model}")
    
    # Test embedding extraction
    print("\n2. Testing embedding extraction...")
    test_texts = ["This is a test tweet.", "Another test tweet about rumors."]
    embeddings = extract_sentence_embeddings(test_texts, model)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Expected: (2, 384)")
    
    # Test feature dimension breakdown
    print("\n3. Feature dimension breakdown:")
    breakdown = get_feature_dimension_breakdown()
    for key, value in breakdown.items():
        print(f"   {key}: {value}")
    
    print("\nModule test completed successfully!")