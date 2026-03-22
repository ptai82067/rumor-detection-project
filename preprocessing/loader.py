#!/usr/bin/env python3
"""
Data loader for PHEME dataset.

This module provides functionality to traverse all events, labels, and threads
in the PHEME dataset structure.
"""

import os
from pathlib import Path
from typing import Generator, Tuple


def load_dataset_paths(data_dir: str = "data/raw/pheme") -> Generator[Tuple[str, int, str], None, None]:
    """
    Traverse all events, labels, and threads in the PHEME dataset.
    
    Args:
        data_dir (str): Path to the PHEME dataset directory
        
    Yields:
        Tuple[str, int, str]: (event_id, label, thread_path)
            - event_id: Name of the event (e.g., 'charliehebdo')
            - label: 1 for rumours, 0 for non-rumours
            - thread_path: Full path to the thread directory
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    
    # Use a set to track processed threads to avoid duplicates
    processed_threads = set()
    
    # Iterate through all event directories
    for event_path in data_path.iterdir():
        if not event_path.is_dir():
            continue
            
        event_id = event_path.name
        print(f"Processing event: {event_id}")
        
        # Iterate through rumours and non-rumours
        for label_dir in event_path.iterdir():
            if not label_dir.is_dir():
                continue
                
            # Map directory names to labels
            if label_dir.name == "rumours":
                label = 1
            elif label_dir.name == "non-rumours":
                label = 0
            else:
                continue  # Skip unknown directories
                
            # Iterate through all thread directories
            for thread_path in label_dir.iterdir():
                if thread_path.is_dir():
                    # Create a unique key for this thread
                    thread_key = f"{event_id}_{label_dir.name}_{thread_path.name}"
                    
                    # Only yield if we haven't processed this thread before
                    if thread_key not in processed_threads:
                        processed_threads.add(thread_key)
                        yield event_id, label, str(thread_path)


def get_thread_info(thread_path: str) -> Tuple[str, str]:
    """
    Extract thread information from thread path.
    
    Args:
        thread_path (str): Path to the thread directory
        
    Returns:
        Tuple[str, str]: (thread_id, thread_path)
    """
    thread_id = Path(thread_path).name
    return thread_id, thread_path


if __name__ == "__main__":
    # Test the loader
    print("Testing data loader...")
    count = 0
    for event_id, label, thread_path in load_dataset_paths():
        count += 1
        if count <= 5:  # Show first 5 entries
            print(f"Event: {event_id}, Label: {label}, Thread: {thread_path}")
        if count > 100:  # Limit output for testing
            break
    print(f"Total threads processed: {count}")