#!/usr/bin/env python3
"""
FIRST Baseline Model for Rumor Detection using PHEME dataset
Event-based split: train on 4 events, test on 1 event
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("FIRST Baseline Model for Rumor Detection")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    df = pd.read_csv('data/processed/pheme_features.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Event-based split
    print("\n2. Event-based split...")
    events = df['event_id'].unique()
    print(f"Available events: {events}")
    print(f"Event counts:\n{df['event_id'].value_counts()}")
    
    # Use 4 largest events for training, smallest for testing
    train_events = ['charliehebdo', 'sydneysiege', 'ferguson', 'ottawashooting']
    test_event = 'germanwings-crash'
    
    train_df = df[df['event_id'].isin(train_events)]
    test_df = df[df['event_id'] == test_event]
    
    print(f"\nTraining events: {train_events}")
    print(f"Test event: {test_event}")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Step 3: Feature construction
    print("\n3. Feature construction...")
    
    # Text features (TF-IDF)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Propagation features
    propagation_features = ['depth', 'children_count', 'thread_size', 
                          'first_reply_time_seconds', 'reply_speed_per_hour', 'max_depth']
    
    # Prepare data
    X_train_text = train_df['text']
    X_train_prop = train_df[propagation_features].fillna(0)
    y_train = train_df['label']
    
    X_test_text = test_df['text']
    X_test_prop = test_df[propagation_features].fillna(0)
    y_test = test_df['label']
    
    # Fit TF-IDF on training text
    X_train_text_tfidf = tfidf.fit_transform(X_train_text)
    X_test_text_tfidf = tfidf.transform(X_test_text)
    
    # Combine features
    X_train = np.hstack([X_train_text_tfidf.toarray(), X_train_prop.values])
    X_test = np.hstack([X_test_text_tfidf.toarray(), X_test_prop.values])
    
    print(f"Training feature shape: {X_train.shape}")
    print(f"Test feature shape: {X_test.shape}")
    
    # Step 4: Model training
    print("\n4. Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Step 5: Evaluation
    print("\n5. Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Training events: {train_events}")
    print(f"Test event: {test_event}")
    print(f"Training size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Feature shape (X_train): {X_train.shape}")
    print(f"Feature shape (X_test): {X_test.shape}")
    print("\nPerformance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 60)

if __name__ == "__main__":
    main()