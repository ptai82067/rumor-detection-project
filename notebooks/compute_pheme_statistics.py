#!/usr/bin/env python3
"""
Compute PHEME dataset statistics directly (standalone script).
Outputs CSV and MD files to results/ directory.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sys

# Configuration
DATA_DIR = Path('data/raw/pheme')
if not DATA_DIR.exists():
    DATA_DIR = Path('d:/rumor-detection-project/data/raw/pheme')

OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {'rumours': 1, 'non-rumours': 0}
LABEL_NAMES = {1: 'Rumor', 0: 'Non-Rumor'}

def parse_timestamp(ts_str):
    if ts_str is None:
        return None
    try:
        return datetime.strptime(ts_str, '%a %b %d %H:%M:%S +0000 %Y')
    except ValueError:
        try:
            return datetime.strptime(ts_str, '%a %b %d %H:%M:%S %z %Y')
        except ValueError:
            return None

print("=" * 60)
print("  PHEME DATASET STATISTICS - COMPUTING...")
print("=" * 60)

# Data structures
tweets = []
threads_info = []
users_set = set()
thread_counter = 0
parse_errors = 0

for event_path in sorted(DATA_DIR.iterdir()):
    if not event_path.is_dir():
        continue
    event_id = event_path.name

    for label_dir_name in ['rumours', 'non-rumours']:
        label_dir = event_path / label_dir_name
        if not label_dir.exists():
            continue

        label = LABEL_MAP[label_dir_name]

        for thread_path in sorted(label_dir.iterdir()):
            if not thread_path.is_dir():
                continue

            thread_id = thread_path.name
            thread_counter += 1

            # Parse source tweet
            source_dir = thread_path / 'source-tweet'
            source_tweet_data = None
            source_tweet_user_id = None
            source_tweet_time = None

            if source_dir.exists():
                json_files = list(source_dir.glob('*.json'))
                if json_files:
                    try:
                        with open(json_files[0], 'r', encoding='utf-8') as f:
                            source_tweet_data = json.load(f)
                        source_tweet_user_id = source_tweet_data.get('user', {}).get('id_str')
                        source_tweet_time = source_tweet_data.get('created_at')

                        tweets.append({
                            'tweet_id': source_tweet_data.get('id_str'),
                            'thread_id': thread_id,
                            'event_id': event_id,
                            'label': label,
                            'is_source': True,
                            'user_id': source_tweet_user_id,
                            'created_at': source_tweet_time,
                        })
                        if source_tweet_user_id:
                            users_set.add(source_tweet_user_id)
                    except Exception:
                        parse_errors += 1

            # Parse reactions
            reactions_dir = thread_path / 'reactions'
            reaction_count = 0

            if reactions_dir.exists():
                for json_file in sorted(reactions_dir.glob('*.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            reaction_data = json.load(f)

                        reaction_user_id = reaction_data.get('user', {}).get('id_str')
                        reaction_time = reaction_data.get('created_at')

                        tweets.append({
                            'tweet_id': reaction_data.get('id_str'),
                            'thread_id': thread_id,
                            'event_id': event_id,
                            'label': label,
                            'is_source': False,
                            'user_id': reaction_user_id,
                            'created_at': reaction_time,
                        })

                        if reaction_user_id:
                            users_set.add(reaction_user_id)

                        reaction_count += 1
                    except Exception:
                        parse_errors += 1

            threads_info.append({
                'thread_id': thread_id,
                'event_id': event_id,
                'label': label,
                'reaction_count': reaction_count,
            })

print(f"Total threads processed: {thread_counter}")
print(f"Total tweets parsed: {len(tweets)}")
print(f"Unique users: {len(users_set)}")
print(f"Parse errors: {parse_errors}")

# Convert to DataFrames
df_tweets = pd.DataFrame(tweets)
df_threads = pd.DataFrame(threads_info)

# ========== COMPUTE ALL STATISTICS ==========

# 1. Basic counts
num_events = df_threads['event_id'].nunique()
num_threads = df_threads['thread_id'].nunique()
num_tweets = len(df_tweets)
num_source_tweets = df_tweets[df_tweets['is_source'] == True].shape[0]
num_reply_tweets = df_tweets[df_tweets['is_source'] == False].shape[0]
num_users = len(users_set)

# 2. Label distribution
rumor_threads = df_threads[df_threads['label'] == 1]
nonrumor_threads = df_threads[df_threads['label'] == 0]
num_rumor = len(rumor_threads)
num_nonrumor = len(nonrumor_threads)
rumor_pct = num_rumor / num_threads * 100
nonrumor_pct = num_nonrumor / num_threads * 100

# 3. Thread size statistics
thread_sizes = df_threads['reaction_count'].values
avg_thread_size = np.mean(thread_sizes)
median_thread_size = np.median(thread_sizes)
max_thread_size = np.max(thread_sizes)
min_thread_size = np.min(thread_sizes)
std_thread_size = np.std(thread_sizes)

# 4. Conversation depth
max_depth = 0
depth_sum = 0
depth_count = 0

for event_path in sorted(DATA_DIR.iterdir()):
    if not event_path.is_dir():
        continue
    for label_dir_name in ['rumours', 'non-rumours']:
        label_dir = event_path / label_dir_name
        if not label_dir.exists():
            continue
        for thread_path in sorted(label_dir.iterdir()):
            if not thread_path.is_dir():
                continue

            reply_parents = {}
            source_id = None

            source_dir = thread_path / 'source-tweet'
            if source_dir.exists():
                json_files = list(source_dir.glob('*.json'))
                if json_files:
                    try:
                        with open(json_files[0], 'r', encoding='utf-8') as f:
                            src = json.load(f)
                        source_id = src.get('id_str')
                    except:
                        pass

            reactions_dir = thread_path / 'reactions'
            if reactions_dir.exists():
                for json_file in reactions_dir.glob('*.json'):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            rxn = json.load(f)
                        reply_id = rxn.get('id_str')
                        parent_id = rxn.get('in_reply_to_status_id_str')
                        if reply_id and parent_id:
                            reply_parents[reply_id] = parent_id
                    except:
                        pass

            if source_id:
                for reply_id in reply_parents:
                    depth = 0
                    current = reply_id
                    visited = set()
                    while current in reply_parents and current not in visited:
                        visited.add(current)
                        current = reply_parents[current]
                        depth += 1
                        if depth > 100:
                            break
                    depth_sum += depth
                    depth_count += 1
                    if depth > max_depth:
                        max_depth = depth

avg_depth = depth_sum / depth_count if depth_count > 0 else 0

# 5. User statistics
user_tweet_counts = df_tweets['user_id'].value_counts()
avg_tweets_per_user = user_tweet_counts.mean()
max_tweets_per_user = user_tweet_counts.max()

# 6. Temporal statistics
df_tweets['parsed_time'] = df_tweets['created_at'].apply(parse_timestamp)
valid_times = df_tweets['parsed_time'].dropna()

if len(valid_times) > 0:
    earliest = valid_times.min()
    latest = valid_times.max()
    coverage_days = (latest - earliest).days
else:
    earliest = None
    latest = None
    coverage_days = 0

# ========== BUILD SUMMARY TABLE ==========

summary_data = [
    ('Number of Events', str(num_events)),
    ('Number of Conversation Threads', str(num_threads)),
    ('Number of Tweets', str(num_tweets)),
    ('Number of Source Tweets', str(num_source_tweets)),
    ('Number of Reply Tweets', str(num_reply_tweets)),
    ('Number of Users', str(num_users)),
    ('', ''),
    ('Number of Rumor Threads', f'{num_rumor} ({rumor_pct:.2f}%)'),
    ('Number of Non-Rumor Threads', f'{num_nonrumor} ({nonrumor_pct:.2f}%)'),
    ('Rumor Percentage', f'{rumor_pct:.2f}%'),
    ('Non-Rumor Percentage', f'{nonrumor_pct:.2f}%'),
    ('', ''),
    ('Average Thread Size (replies)', f'{avg_thread_size:.2f}'),
    ('Median Thread Size (replies)', f'{median_thread_size:.0f}'),
    ('Maximum Thread Size (replies)', str(max_thread_size)),
    ('Minimum Thread Size (replies)', str(min_thread_size)),
    ('', ''),
    ('Maximum Conversation Depth', str(max_depth)),
    ('Average Conversation Depth', f'{avg_depth:.2f}'),
    ('', ''),
    ('Average Tweets per User', f'{avg_tweets_per_user:.2f}'),
    ('Maximum Tweets per User', str(max_tweets_per_user)),
    ('', ''),
]

if earliest is not None:
    summary_data.append(('Earliest Tweet Timestamp', earliest.strftime('%Y-%m-%d %H:%M:%S')))
    summary_data.append(('Latest Tweet Timestamp', latest.strftime('%Y-%m-%d %H:%M:%S')))
    summary_data.append(('Data Coverage Duration (days)', str(coverage_days)))
else:
    summary_data.append(('Earliest Tweet Timestamp', 'N/A (no valid timestamps)'))
    summary_data.append(('Latest Tweet Timestamp', 'N/A (no valid timestamps)'))
    summary_data.append(('Data Coverage Duration (days)', 'N/A'))

df_summary = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
df_summary_clean = df_summary[df_summary['Metric'] != ''].copy()

# ========== SAVE CSV ==========

csv_path = OUTPUT_DIR / 'pheme_dataset_statistics.csv'
df_summary_clean.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nCSV saved to: {csv_path}")

# ========== SAVE MARKDOWN ==========

event_names_list = ', '.join([e.replace('-', ' ').title() for e in sorted(df_threads['event_id'].unique())])

if earliest is not None:
    temporal_desc = (
        f"The dataset spans from {earliest.strftime('%B %d, %Y')} to "
        f"{latest.strftime('%B %d, %Y')}, covering a period of {coverage_days} days. "
    )
else:
    temporal_desc = ""

md_lines = []
md_lines.append("# PHEME Dataset Statistics\n")
md_lines.append("## Overview\n")
md_lines.append(f"The PHEME dataset is a collection of Twitter conversations gathered during five breaking news events: {event_names_list}.\n")
md_lines.append("## Summary Table\n")
md_lines.append("| Metric | Value |\n| --- | --- |\n")

for _, row in df_summary_clean.iterrows():
    metric = row['Metric']
    value = row['Value']
    if pd.notna(metric) and str(metric).strip():
        md_lines.append(f"| {metric} | {value} |\n")

md_lines.append("\n## Key Observations\n\n")
md_lines.append(f"The PHEME dataset comprises {num_events} distinct breaking news events, containing a total of {num_threads} conversation threads and {num_tweets:,} tweets. ")
md_lines.append(f"Among these, {num_source_tweets:,} are source tweets initiating discussions, and {num_reply_tweets:,} are replies forming the conversation threads. ")
md_lines.append(f"The dataset includes {num_users:,} unique users who participated in these discussions. ")
md_lines.append(f"Regarding rumor distribution, {num_rumor} threads ({rumor_pct:.2f}%) are labeled as rumors, while {num_nonrumor} threads ({nonrumor_pct:.2f}%) are non-rumors. ")
md_lines.append(f"This relatively balanced distribution makes the dataset suitable for binary classification tasks. ")
md_lines.append(f"The conversation threads vary significantly in size, with an average of {avg_thread_size:.2f} replies per thread and a maximum of {max_thread_size} replies. ")
md_lines.append(f"The maximum conversation depth reaches {max_depth} levels, indicating complex discussion structures, particularly in controversial rumors. ")
md_lines.append(temporal_desc)
md_lines.append(f"User participation is diverse, with an average of {avg_tweets_per_user:.2f} tweets per user, highlighting varied engagement levels across the dataset. ")
md_lines.append("\n---\n*This statistical summary was automatically generated from the raw PHEME dataset for use in the graduation thesis.*\n")

md_path = OUTPUT_DIR / 'pheme_dataset_statistics.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.writelines(md_lines)
print(f"Markdown saved to: {md_path}")

# ========== PRINT SUMMARY ==========

print("\n" + "=" * 60)
print("  PHEME DATASET STATISTICS - COMPLETE")
print("=" * 60)
print(f"\n{'Metric':<40} {'Value':<20}")
print("-" * 60)
for _, row in df_summary_clean.iterrows():
    metric = row['Metric']
    value = row['Value']
    if pd.notna(metric) and str(metric).strip():
        print(f"{str(metric):<40} {str(value):<20}")
print()
print("Output files:")
print(f"  1. {csv_path}")
print(f"  2. {md_path}")
print(f"  3. notebooks/pheme_dataset_statistics.ipynb")