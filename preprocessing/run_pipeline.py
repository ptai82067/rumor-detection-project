#!/usr/bin/env python3
"""
Main preprocessing pipeline runner.

This module orchestrates the complete preprocessing pipeline for the PHEME dataset,
including data loading, parsing, normalization, and CSV export.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Import our modules
from .loader import load_dataset_paths
from .parser import parse_thread
from .normalizer import normalize_dataset


def run_preprocessing_pipeline(data_dir: str = "data/raw/pheme", 
                              output_path: str = "data/processed/pheme_clean.csv") -> None:
    """
    Execute the complete preprocessing pipeline on the FULL dataset.
    
    Process:
    - Loop through all events and threads using loader
    - Parse each thread (source + reactions)
    - Normalize all tweets
    - Append to master list
    - Export to CSV
    
    Args:
        data_dir (str): Path to the raw PHEME dataset
        output_path (str): Path where the processed CSV will be saved
    """
    print("=" * 60)
    print("RUMOR DETECTION PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input directory: {data_dir}")
    print(f"Output file: {output_path}")
    print("=" * 60)
    
    # Initialize master list to store all normalized records
    all_records = []
    
    # Track processing statistics
    total_threads = 0
    total_tweets = 0
    processed_events = set()
    
    try:
        # Execute the pipeline
        for event_id, label, thread_path in load_dataset_paths(data_dir):
            total_threads += 1
            processed_events.add(event_id)
            
            # Get thread ID from path
            thread_id = Path(thread_path).name
            
            # Log current processing status
            print(f"Processing - Event: {event_id}, Thread: {thread_id}, Label: {label}")
            
            # Parse the thread (source tweet + reactions)
            source_tweet, reactions = parse_thread(thread_path)
            
            # Skip threads with no source tweet
            if not source_tweet:
                print(f"  Warning: No source tweet found for thread {thread_id}")
                continue
            
            # Normalize all tweets in the thread
            normalized_records = normalize_dataset(
                source_tweet, 
                reactions, 
                event_id, 
                thread_id, 
                label
            )
            
            # Add to master list
            all_records.extend(normalized_records)
            total_tweets += len(normalized_records)
            
            # Log progress every 100 threads
            if total_threads % 100 == 0:
                print(f"  Progress: {total_threads} threads processed, {total_tweets} tweets normalized")
    
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        raise
    
    # Convert to DataFrame
    print(f"\nPipeline completed!")
    print(f"Total threads processed: {total_threads}")
    print(f"Total tweets normalized: {total_tweets}")
    print(f"Events processed: {len(processed_events)} - {', '.join(sorted(processed_events))}")
    
    if not all_records:
        print("No records to export!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Validate data
    print("\n" + "=" * 40)
    print("DATA VALIDATION")
    print("=" * 40)
    
    # 1. Print total number of rows
    print(f"Total number of rows: {len(df)}")
    
    # 2. Show label distribution
    print("\nLabel distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Label {label}: {count} tweets ({percentage:.2f}%)")
    
    # 3. Show missing values count
    print("\nMissing values:")
    missing_values = df.isnull().sum()
    for column, count in missing_values.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"  {column}: {count} missing ({percentage:.2f}%)")
        else:
            print(f"  {column}: 0 missing")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV
    print(f"\nExporting to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print(f"CSV export completed successfully!")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Show sample rows
    print("\n" + "=" * 40)
    print("SAMPLE DATA (First 5 rows)")
    print("=" * 40)
    print(df.head())
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the pipeline
    run_preprocessing_pipeline()