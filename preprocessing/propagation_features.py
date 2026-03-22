"""
Propagation-based feature engineering for rumor detection.

This module extracts structural and temporal features from tweet threads
to capture propagation patterns that are indicative of rumors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PropagationFeatureExtractor:
    """Extract propagation-based features from tweet threads."""
    
    def __init__(self):
        self.thread_structure_cache = {}
        self.thread_features_cache = {}
    
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Sort by thread_id and time for efficient processing
        df = df.sort_values(['thread_id', 'time']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} tweets across {df['thread_id'].nunique()} threads")
        return df
    
    def remove_duplicate_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate post_ids, keeping the earliest occurrence."""
        logger.info("Checking for duplicate post_ids...")
        
        # Count duplicates
        original_count = len(df)
        duplicate_mask = df.duplicated('post_id', keep=False)
        num_duplicates = duplicate_mask.sum()
        
        if num_duplicates == 0:
            logger.info("No duplicate post_ids found.")
            return df
        
        logger.warning(f"Found {num_duplicates} duplicate post_ids out of {original_count} total tweets")
        
        # Check for conflicting data in duplicates
        duplicates_df = df[duplicate_mask].copy()
        duplicate_post_ids = duplicates_df['post_id'].unique()
        
        conflicting_data = []
        for post_id in duplicate_post_ids:
            post_rows = duplicates_df[duplicates_df['post_id'] == post_id]
            
            # Check if any columns have conflicting values
            for col in ['user_id', 'text', 'event_id', 'thread_id']:
                if post_rows[col].nunique() > 1:
                    conflicting_data.append({
                        'post_id': post_id,
                        'column': col,
                        'values': post_rows[col].tolist()
                    })
        
        if conflicting_data:
            logger.warning(f"Found {len(conflicting_data)} posts with conflicting data in duplicates:")
            for conflict in conflicting_data[:5]:  # Show first 5 conflicts
                logger.warning(f"  Post {conflict['post_id']}, column '{conflict['column']}': {conflict['values']}")
            if len(conflicting_data) > 5:
                logger.warning(f"  ... and {len(conflicting_data) - 5} more conflicts")
        
        # Remove duplicates, keeping the earliest occurrence based on time
        # Sort by post_id and time to ensure consistent behavior
        df_sorted = df.sort_values(['post_id', 'time'])
        df_clean = df_sorted.drop_duplicates(subset=['post_id'], keep='first')
        
        final_count = len(df_clean)
        removed_count = original_count - final_count
        
        logger.info(f"Removed {removed_count} duplicate posts")
        logger.info(f"Final dataset: {final_count} tweets across {df_clean['thread_id'].nunique()} threads")
        
        # Final validation
        final_duplicates = df_clean.duplicated('post_id').sum()
        if final_duplicates > 0:
            logger.error(f"ERROR: Still found {final_duplicates} duplicate post_ids after cleaning!")
            raise ValueError("Duplicate removal failed")
        else:
            logger.info("✅ No duplicate post_ids remain after cleaning")
        
        return df_clean
    
    def build_thread_structure(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """Build parent-child relationships for each thread."""
        logger.info("Building thread structure...")
        
        thread_groups = df.groupby('thread_id')
        thread_structures = {}
        
        for thread_id, thread_data in thread_groups:
            # Create mapping of post_id to reply_to
            post_to_parent = dict(zip(thread_data['post_id'], thread_data['reply_to']))
            
            # Find source tweets (where reply_to is NaN)
            source_posts = thread_data[thread_data['reply_to'].isna()]['post_id'].tolist()
            
            # Build parent-child relationships
            parent_to_children = {}
            for post_id, parent_id in post_to_parent.items():
                if pd.isna(parent_id):
                    continue
                if parent_id not in parent_to_children:
                    parent_to_children[parent_id] = []
                parent_to_children[parent_id].append(post_id)
            
            thread_structures[thread_id] = {
                'post_to_parent': post_to_parent,
                'parent_to_children': parent_to_children,
                'source_posts': source_posts,
                'post_times': dict(zip(thread_data['post_id'], thread_data['time']))
            }
        
        self.thread_structure_cache = thread_structures
        logger.info(f"Built structure for {len(thread_structures)} threads")
        return thread_structures
    
    def calculate_depth(self, thread_id: int, post_id: int, 
                       post_to_parent: Dict[int, Optional[int]], 
                       depth_cache: Dict[int, int]) -> int:
        """Calculate depth of a post in the thread tree."""
        if post_id in depth_cache:
            return depth_cache[post_id]
        
        parent_id = post_to_parent.get(post_id)
        
        if pd.isna(parent_id) or parent_id not in post_to_parent:
            # This is a source post
            depth = 0
        else:
            # Recursively calculate parent depth
            depth = self.calculate_depth(thread_id, parent_id, post_to_parent, depth_cache) + 1
        
        depth_cache[post_id] = depth
        return depth
    
    def extract_node_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract node-level (tweet-level) features."""
        logger.info("Extracting node-level features...")
        
        # Initialize feature columns
        df_features = df.copy()
        df_features['depth'] = 0
        df_features['children_count'] = 0
        df_features['time_since_source'] = 0
        df_features['is_source'] = False
        
        # Process each thread
        for thread_id, thread_data in df_features.groupby('thread_id'):
            if thread_id not in self.thread_structure_cache:
                continue
                
            thread_structure = self.thread_structure_cache[thread_id]
            post_to_parent = thread_structure['post_to_parent']
            parent_to_children = thread_structure['parent_to_children']
            post_times = thread_structure['post_times']
            source_posts = thread_structure['source_posts']
            
            # Calculate depth for each post in this thread
            depth_cache = {}
            for post_id in post_to_parent.keys():
                depth = self.calculate_depth(thread_id, post_id, post_to_parent, depth_cache)
                
                # Update the dataframe
                mask = (df_features['thread_id'] == thread_id) & (df_features['post_id'] == post_id)
                df_features.loc[mask, 'depth'] = depth
                df_features.loc[mask, 'is_source'] = post_id in source_posts
                
                # Calculate children count
                children_count = len(parent_to_children.get(post_id, []))
                df_features.loc[mask, 'children_count'] = children_count
                
                # Calculate time since source
                if post_id in source_posts:
                    time_since_source = 0
                else:
                    source_time = min(post_times[sp] for sp in source_posts)
                    time_since_source = (post_times[post_id] - source_time).total_seconds()
                
                df_features.loc[mask, 'time_since_source'] = time_since_source
        
        logger.info("Node-level features extracted")
        return df_features
    
    def extract_thread_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract thread-level features and merge them to the main dataframe."""
        logger.info("Extracting thread-level features...")
        
        thread_features = []
        
        for thread_id, thread_data in df.groupby('thread_id'):
            if thread_id not in self.thread_structure_cache:
                continue
                
            thread_structure = self.thread_structure_cache[thread_id]
            post_times = thread_structure['post_times']
            source_posts = thread_structure['source_posts']
            
            # Thread size
            thread_size = len(thread_data)
            
            # Time duration
            min_time = min(post_times.values())
            max_time = max(post_times.values())
            duration_seconds = (max_time - min_time).total_seconds()
            
            # First reply time
            source_time = min(post_times[sp] for sp in source_posts)
            non_source_times = [t for pid, t in post_times.items() if pid not in source_posts]
            first_reply_time = 0
            if non_source_times:
                first_reply_time = (min(non_source_times) - source_time).total_seconds()
            
            # Reply speed (replies per hour)
            reply_count = thread_size - len(source_posts)
            reply_speed = 0
            if duration_seconds > 0:
                reply_speed = reply_count / (duration_seconds / 3600)  # per hour
            
            # Max depth
            max_depth = max(thread_data['depth']) if 'depth' in thread_data.columns else 0
            
            thread_features.append({
                'thread_id': thread_id,
                'thread_size': thread_size,
                'thread_duration_hours': duration_seconds / 3600,
                'first_reply_time_seconds': first_reply_time,
                'reply_speed_per_hour': reply_speed,
                'max_depth': max_depth
            })
        
        # Create thread features dataframe
        thread_df = pd.DataFrame(thread_features)
        
        # Merge thread features to main dataframe
        df_with_thread_features = df.merge(thread_df, on='thread_id', how='left')
        
        logger.info("Thread-level features extracted and merged")
        return df_with_thread_features
    
    def extract_all_features(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Complete feature extraction pipeline."""
        logger.info("Starting propagation feature extraction pipeline...")
        
        # Step 1: Load and preprocess data
        df = self.load_and_preprocess_data(input_file)
        
        # Step 2: Remove duplicate post_ids (CRITICAL STEP)
        df_clean = self.remove_duplicate_posts(df)
        
        # Step 3: Build thread structure
        self.build_thread_structure(df_clean)
        
        # Step 4: Extract node-level features
        df_with_node_features = self.extract_node_level_features(df_clean)
        
        # Step 5: Extract thread-level features
        df_final = self.extract_thread_level_features(df_with_node_features)
        
        # Step 6: Save to output file
        logger.info(f"Saving features to {output_file}")
        df_final.to_csv(output_file, index=False)
        
        # Step 7: Validation
        self.validate_features(df_final)
        
        logger.info("Feature extraction pipeline completed successfully!")
        return df_final
    
    def validate_features(self, df: pd.DataFrame):
        """Validate the extracted features."""
        logger.info("Validating extracted features...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found {missing_values.sum()} missing values")
            print("Missing values by column:")
            print(missing_values[missing_values > 0])
        else:
            logger.info("No missing values found")
        
        # Basic statistics
        print("\nFeature statistics:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of threads: {df['thread_id'].nunique()}")
        print(f"Average thread size: {df.groupby('thread_id').size().mean():.2f}")
        print(f"Max depth: {df['max_depth'].max()}")
        print(f"Average reply speed: {df['reply_speed_per_hour'].mean():.2f} replies/hour")
        
        # Sample rows
        print("\nSample rows:")
        print(df.head())


def main():
    """Main function to run the feature extraction pipeline."""
    extractor = PropagationFeatureExtractor()
    
    input_file = "data/processed/pheme_clean.csv"
    output_file = "data/processed/pheme_features.csv"
    
    try:
        df = extractor.extract_all_features(input_file, output_file)
        print(f"\n✅ Successfully created {output_file}")
        print(f"✅ Extracted features for {len(df)} tweets")
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        raise


if __name__ == "__main__":
    main()