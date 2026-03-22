#!/usr/bin/env python3
"""
JSON parser for PHEME dataset.

This module provides functionality to read and parse JSON files containing
tweets from source-tweet and reactions directories.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_tweet_json(json_path: str) -> Optional[Dict]:
    """
    Parse a single tweet JSON file and extract required fields.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        Optional[Dict]: Parsed tweet data or None if parsing fails
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            tweet_data = json.load(f)
            
        # Extract required fields with safe handling
        parsed_tweet = {
            'post_id': tweet_data.get('id_str'),
            'user_id': tweet_data.get('user', {}).get('id_str'),
            'text': tweet_data.get('text'),
            'time': tweet_data.get('created_at'),
            'reply_to': tweet_data.get('in_reply_to_status_id_str')
        }
        
        # Validate that we have essential fields
        if not parsed_tweet['post_id'] or not parsed_tweet['text']:
            print(f"Warning: Missing essential fields in {json_path}")
            return None
            
        return parsed_tweet
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {json_path}: {e}")
        return None


def parse_source_tweet(thread_path: str) -> Optional[Dict]:
    """
    Parse the source tweet from a thread.
    
    Args:
        thread_path (str): Path to the thread directory
        
    Returns:
        Optional[Dict]: Parsed source tweet data or None if not found/parsed
    """
    source_dir = Path(thread_path) / "source-tweet"
    
    if not source_dir.exists():
        print(f"Source tweet directory not found: {source_dir}")
        return None
    
    # Find the JSON file in source-tweet directory
    json_files = list(source_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in source-tweet directory: {source_dir}")
        return None
    
    if len(json_files) > 1:
        print(f"Multiple JSON files found in source-tweet directory: {source_dir}")
        # Use the first one
        json_file = json_files[0]
    else:
        json_file = json_files[0]
    
    return parse_tweet_json(str(json_file))


def parse_reactions(thread_path: str) -> List[Dict]:
    """
    Parse all reaction tweets from a thread.
    
    Args:
        thread_path (str): Path to the thread directory
        
    Returns:
        List[Dict]: List of parsed reaction tweets
    """
    reactions_dir = Path(thread_path) / "reactions"
    reactions = []
    
    if not reactions_dir.exists():
        print(f"Reactions directory not found: {reactions_dir}")
        return reactions
    
    # Find all JSON files in reactions directory
    json_files = list(reactions_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in reactions directory: {reactions_dir}")
        return reactions
    
    # Parse each reaction tweet
    for json_file in json_files:
        reaction = parse_tweet_json(str(json_file))
        if reaction:
            reactions.append(reaction)
    
    return reactions


def parse_thread(thread_path: str) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Parse a complete thread including source tweet and all reactions.
    
    Args:
        thread_path (str): Path to the thread directory
        
    Returns:
        Tuple[Optional[Dict], List[Dict]]: (source_tweet, reactions)
    """
    source_tweet = parse_source_tweet(thread_path)
    reactions = parse_reactions(thread_path)
    
    return source_tweet, reactions


if __name__ == "__main__":
    # Test the parser with a sample thread
    test_thread = "data/raw/pheme/charliehebdo/rumours/552783238415265792"
    
    print(f"Testing parser with thread: {test_thread}")
    source, reactions = parse_thread(test_thread)
    
    if source:
        print(f"Source tweet: {source}")
    else:
        print("No source tweet found")
    
    print(f"Number of reactions: {len(reactions)}")
    for i, reaction in enumerate(reactions[:3]):  # Show first 3 reactions
        print(f"Reaction {i+1}: {reaction}")