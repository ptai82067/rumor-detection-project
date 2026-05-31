#!/usr/bin/env python3
"""Quick PHEME dataset statistics - counts directories and parses minimal JSON fields."""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

DATA_DIR = Path('data/raw/pheme')
if not DATA_DIR.exists():
    DATA_DIR = Path('d:/rumor-detection-project/data/raw/pheme')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)

events = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"Events: {events}")

rumor_count = 0
nonrumor_count = 0
thread_sizes = []
event_thread_counts = Counter()
users = set()
all_times = []

for event in events:
    for label_dir_name in ['rumours', 'non-rumours']:
        label_dir = DATA_DIR / event / label_dir_name
        if not label_dir.exists():
            continue
        threads = sorted([t.name for t in label_dir.iterdir() if t.is_dir()])
        event_thread_counts[event] += len(threads)
        
        for thread_id in threads:
            thread_path = label_dir / thread_id
            source_dir = thread_path / 'source-tweet'
            reactions_dir = thread_path / 'reactions'
            
            source_files = list(source_dir.glob('*.json')) if source_dir.exists() else []
            reply_files = list(reactions_dir.glob('*.json')) if reactions_dir.exists() else []
            num_replies = len(reply_files)
            thread_sizes.append(num_replies)
            
            if label_dir_name == 'rumours':
                rumor_count += 1
            else:
                nonrumor_count += 1
            
            # Parse source user/time
            if source_files:
                try:
                    with open(source_files[0], 'r', encoding='utf-8') as f:
                        src = json.load(f)
                    uid = src.get('user', {}).get('id_str')
                    if uid:
                        users.add(uid)
                    ct = src.get('created_at')
                    if ct:
                        all_times.append(ct)
                except Exception:
                    pass
            
            # Parse reply user/time
            for rf in reply_files:
                try:
                    with open(rf, 'r', encoding='utf-8') as f:
                        rxn = json.load(f)
                    uid = rxn.get('user', {}).get('id_str')
                    if uid:
                        users.add(uid)
                    ct = rxn.get('created_at')
                    if ct:
                        all_times.append(ct)
                except Exception:
                    pass

total_threads = rumor_count + nonrumor_count
total_tweets = total_threads + sum(thread_sizes)

# Timestamp parsing
def parse_ts(s):
    if not s:
        return None
    for fmt in ['%a %b %d %H:%M:%S +0000 %Y', '%a %b %d %H:%M:%S %z %Y']:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

parsed_times = [parse_ts(t) for t in all_times if t]
parsed_times = [p for p in parsed_times if p]

if parsed_times:
    earliest = min(parsed_times)
    latest = max(parsed_times)
    coverage = (latest - earliest).days
    earliest_str = earliest.strftime('%Y-%m-%d %H:%M:%S')
    latest_str = latest.strftime('%Y-%m-%d %H:%M:%S')
else:
    earliest_str = 'N/A'
    latest_str = 'N/A'
    coverage = 'N/A'

# Thread size stats
mean_size = np.mean(thread_sizes) if thread_sizes else 0
median_size = np.median(thread_sizes) if thread_sizes else 0
max_size = max(thread_sizes) if thread_sizes else 0
min_size = min(thread_sizes) if thread_sizes else 0
std_size = np.std(thread_sizes) if thread_sizes else 0

rumor_pct = rumor_count / total_threads * 100
nonrumor_pct = nonrumor_count / total_threads * 100

# Print results
print("\n" + "=" * 60)
print("  PHEME DATASET STATISTICS - RESULTS")
print("=" * 60)
print(f"\n{'Metric':<40} {'Value':<20}")
print("-" * 60)

rows = [
    ("Number of Events", str(len(events))),
    ("Number of Conversation Threads", str(total_threads)),
    ("Number of Tweets", str(total_tweets)),
    ("Number of Source Tweets", str(total_threads)),
    ("Number of Reply Tweets", str(total_tweets - total_threads)),
    ("Number of Users", str(len(users))),
    ("", ""),
    ("Number of Rumor Threads", f"{rumor_count} ({rumor_pct:.2f}%)"),
    ("Number of Non-Rumor Threads", f"{nonrumor_count} ({nonrumor_pct:.2f}%)"),
    ("Rumor Percentage", f"{rumor_pct:.2f}%"),
    ("Non-Rumor Percentage", f"{nonrumor_pct:.2f}%"),
    ("", ""),
    ("Average Thread Size (replies)", f"{mean_size:.2f}"),
    ("Median Thread Size (replies)", f"{median_size:.0f}"),
    ("Maximum Thread Size (replies)", str(max_size)),
    ("Minimum Thread Size (replies)", str(min_size)),
    ("", ""),
    ("Average Tweets per User", f"{total_tweets/len(users):.2f}" if users else "0"),
    ("Maximum Tweets per User", "[computed per user]" ),
    ("", ""),
    ("Earliest Tweet Timestamp", earliest_str),
    ("Latest Tweet Timestamp", latest_str),
    ("Data Coverage Duration (days)", str(coverage)),
]

for metric, value in rows:
    print(f"{metric:<40} {value:<20}")

# Save CSV
df_summary = pd.DataFrame(rows, columns=['Metric', 'Value'])
df_summary_clean = df_summary[df_summary['Metric'] != ''].copy()
df_summary_clean.to_csv(OUTPUT_DIR / 'pheme_dataset_statistics.csv', index=False, encoding='utf-8-sig')
print(f"\nCSV: {OUTPUT_DIR / 'pheme_dataset_statistics.csv'}")

# Save Markdown
event_names = ', '.join(e.replace('-', ' ').title() for e in events)
if parsed_times:
    temporal = f"The dataset spans from {earliest.strftime('%B %d, %Y')} to {latest.strftime('%B %d, %Y')}, covering a period of {coverage} days. "
else:
    temporal = ""

md = "# PHEME Dataset Statistics\n\n"
md += "## Overview\n\n"
md += f"The PHEME dataset is a collection of Twitter conversations gathered during five breaking news events: {event_names}.\n\n"
md += "## Summary Table\n\n"
md += "| Metric | Value |\n| --- | --- |\n"

for _, row in df_summary_clean.iterrows():
    m, v = row['Metric'], row['Value']
    if pd.notna(m) and str(m).strip():
        md += f"| {m} | {v} |\n"

md += "\n## Key Observations\n\n"
md += f"The PHEME dataset comprises {len(events)} distinct breaking news events, containing a total of {total_threads} conversation threads and {total_tweets:,} tweets. "
md += f"Among these, {total_threads:,} are source tweets initiating discussions, and {total_tweets - total_threads:,} are replies forming the conversation threads. "
md += f"The dataset includes {len(users):,} unique users who participated in these discussions.\n\n"
md += f"Regarding rumor distribution, {rumor_count} threads ({rumor_pct:.2f}%) are labeled as rumors, while {nonrumor_count} threads ({nonrumor_pct:.2f}%) are non-rumors. "
md += "This relatively balanced distribution makes the dataset suitable for binary classification tasks.\n\n"
md += f"The conversation threads vary significantly in size, with an average of {mean_size:.2f} replies per thread and a maximum of {max_size} replies. "
md += f"The median thread size is {median_size:.0f} replies, indicating that most discussions are relatively compact while a few threads contain extensive debates.\n\n"
md += f"{temporal}"
md += f"User participation is diverse, with an average of {total_tweets/len(users):.2f} tweets per user, highlighting varied engagement levels across the dataset.\n\n"
md += "---\n*This statistical summary was automatically generated from the raw PHEME dataset for use in the graduation thesis.*\n"

(OUTPUT_DIR / 'pheme_dataset_statistics.md').write_text(md, encoding='utf-8')
print(f"MD:  {OUTPUT_DIR / 'pheme_dataset_statistics.md'}")
print("\nDONE - All files generated successfully!")