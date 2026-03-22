#!/usr/bin/env python3
"""
Text normalizer for preprocessing.

This module provides functionality to convert raw tweets into structured rows
with a consistent schema for rumor detection.
"""

from typing import Dict, List, Optional


def normalize_source_tweet(source_tweet: Dict, event_id: str, thread_id: str) -> Dict:
    """
    Normalize a source tweet into the required schema.
    
    Args:
        source_tweet (Dict): Parsed source tweet data
        event_id (str): Event identifier
        thread_id (str): Thread identifier (same as source post_id)
        
    Returns:
        Dict: Normalized source tweet record
    """
    return {
        'post_id': source_tweet['post_id'],
        'user_id': source_tweet['user_id'],
        'text': source_tweet['text'],
        'time': source_tweet['time'],
        'event_id': event_id,
        'reply_to': None,  # Source tweets don't have reply_to
        'thread_id': thread_id,
        'label': None  # Label will be set at thread level
    }


def normalize_reaction_tweet(reaction_tweet: Dict, event_id: str, thread_id: str) -> Dict:
    """
    Normalize a reaction tweet into the required schema.
    
    Args:
        reaction_tweet (Dict): Parsed reaction tweet data
        event_id (str): Event identifier
        thread_id (str): Thread identifier (same as source post_id)
        
    Returns:
        Dict: Normalized reaction tweet record
    """
    return {
        'post_id': reaction_tweet['post_id'],
        'user_id': reaction_tweet['user_id'],
        'text': reaction_tweet['text'],
        'time': reaction_tweet['time'],
        'event_id': event_id,
        'reply_to': reaction_tweet['reply_to'],
        'thread_id': thread_id,
        'label': None  # Label will be set at thread level
    }


def normalize_thread(source_tweet: Optional[Dict], reactions: List[Dict], 
                    event_id: str, thread_id: str, label: int) -> List[Dict]:
    """
    Normalize all tweets in a thread (source + reactions) into structured records.
    
    Args:
        source_tweet (Optional[Dict]): Parsed source tweet data
        reactions (List[Dict]): List of parsed reaction tweets
        event_id (str): Event identifier
        thread_id (str): Thread identifier
        label (int): Thread label (1 for rumours, 0 for non-rumours)
        
    Returns:
        List[Dict]: List of normalized tweet records
    """
    normalized_records = []
    
    # Normalize source tweet if it exists
    if source_tweet:
        source_record = normalize_source_tweet(source_tweet, event_id, thread_id)
        source_record['label'] = label  # Set label for source tweet
        normalized_records.append(source_record)
    
    # Normalize all reaction tweets
    for reaction in reactions:
        reaction_record = normalize_reaction_tweet(reaction, event_id, thread_id)
        reaction_record['label'] = label  # Set label for reaction tweets
        normalized_records.append(reaction_record)
    
    return normalized_records


def normalize_dataset(source_tweet: Optional[Dict], reactions: List[Dict], 
                     event_id: str, thread_id: str, label: int) -> List[Dict]:
    """
    Main normalization function that processes a complete thread.
    
    This function serves as the main entry point for normalization.
    
    Args:
        source_tweet (Optional[Dict]): Parsed source tweet data
        reactions (List[Dict]): List of parsed reaction tweets
        event_id (str): Event identifier
        thread_id (str): Thread identifier
        label (int): Thread label (1 for rumours, 0 for non-rumours)
        
    Returns:
        List[Dict]: List of normalized tweet records for the entire thread
    """
    return normalize_thread(source_tweet, reactions, event_id, thread_id, label)


if __name__ == "__main__":
    # Test the normalizer
    test_source = {
        'post_id': '552783238415265792',
        'user_id': '384779793',
        'text': 'Breaking: At least 10 dead...',
        'time': 'Wed Jan 07 11:06:08 +0000 2015',
        'reply_to': None
    }
    
    test_reactions = [
        {
            'post_id': '552787794503143424',
            'user_id': '202572421',
            'text': '@H_E_Samuel The religion of peace strikes again.',
            'time': 'Wed Jan 07 11:24:15 +0000 2015',
            'reply_to': '552783238415265792'
        }
    ]
    
    print("Testing normalizer...")
    normalized = normalize_dataset(
        test_source, 
        test_reactions, 
        'charliehebdo', 
        '552783238415265792', 
        1
    )
    
    for record in normalized:
        print(f"Normalized record: {record}")