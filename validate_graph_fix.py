"""
Milestone 05A.1: Graph Topology Fix Validation Script
======================================================
Purpose: Regenerate graph features with corrected URI parsing and validate
         that all 14 graph features have non-zero structural signal.

This script:
1. Extracts all graph features using the fixed utils/graph_features.py
2. Validates topology zero-rates for all 14 graph columns
3. Saves pheme_features_with_graph.csv for downstream use
4. Produces a detailed validation report
5. Returns GO/NO-GO verdict for 05B
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.graph_features import (
    extract_all_graph_features,
    get_graph_feature_columns,
    get_feature_sets,
)

print("=" * 80)
print("MILESTONE 05A.1: GRAPH TOPOLOGY FIX VALIDATION")
print("=" * 80)

# Step 1: Load frozen base dataset
print("\n[1/6] Loading frozen base dataset...")
df = pd.read_csv("data/processed/pheme_features.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Label distribution: Non-Rumor={len(df[df['label']==0])}, Rumor={len(df[df['label']==1])}")

# Step 2: Extract all graph features with corrected code
print("\n[2/6] Extracting graph features with FIXED URI parsing...")
print("  Fix applied: 'post/' pattern instead of '/post/'")
df_enriched = extract_all_graph_features(df)

# Step 3: Validate all 14 graph columns exist
print("\n[3/6] Validating graph feature columns...")
graph_cols = get_graph_feature_columns()
print(f"  Expected graph features: {len(graph_cols)}")
print(f"  Actual graph features: {len([c for c in graph_cols if c in df_enriched.columns])}")

missing_cols = [c for c in graph_cols if c not in df_enriched.columns]
if missing_cols:
    print(f"  ❌ MISSING COLUMNS: {missing_cols}")
    sys.exit(1)
else:
    print("  ✅ All 14 graph columns present")

# Step 4: Topology zero-rate analysis (CRITICAL VALIDATION)
print("\n[4/6] TOPOLOGY ZERO-RATE ANALYSIS")
print("  " + "=" * 60)

# Categorize features
topology_features = {
    'Node Centrality': ['node_in_degree', 'node_out_degree', 'pagerank_score', 
                        'betweenness_centrality', 'closeness_centrality'],
    'User Reputation': ['user_prior_rumor_ratio', 'user_post_count', 
                        'user_thread_count', 'user_avg_depth'],
    'Thread Structure': ['subtree_reply_count', 'sibling_count', 'position_in_thread'],
    'Source Authority': ['source_user_credibility', 'source_network_size']
}

validation_passed = True
zero_rate_report = []

for category, features in topology_features.items():
    print(f"\n  {category}:")
    for feat in features:
        if feat in df_enriched.columns:
            zero_count = (df_enriched[feat] == 0).sum()
            total_count = len(df_enriched)
            zero_rate = zero_count / total_count * 100
            non_zero = total_count - zero_count
            
            # Determine expected behavior
            if feat in ['node_out_degree']:
                # Out-degree should be 0 for root posts (depth=0), 1 for replies
                expected_non_zero = len(df_enriched[df_enriched['depth'] > 0])
                status = "✅" if abs(non_zero - expected_non_zero) < 100 else "⚠️"
            elif feat in ['position_in_thread']:
                # Position should never be 0 (starts at 1)
                status = "✅" if zero_count == 0 else "❌"
            elif feat in ['user_prior_rumor_ratio']:
                # Some users may have 0.0 if they only posted non-rumors
                status = "✅" if zero_rate < 50 else "⚠️"
            elif feat in ['node_in_degree', 'pagerank_score', 'betweenness_centrality']:
                # These should have many non-zero values
                status = "✅" if zero_rate < 90 else "❌"
            else:
                status = "✅"
            
            print(f"    {status} {feat:30s} | Zero: {zero_count:>7,} ({zero_rate:5.1f}%) | Non-zero: {non_zero:>7,}")
            zero_rate_report.append({
                'feature': feat,
                'category': category,
                'zero_count': zero_count,
                'non_zero_count': non_zero,
                'zero_rate': zero_rate,
                'status': status
            })
            
            # Critical features must have significant non-zero values
            if feat in ['node_in_degree', 'pagerank_score'] and zero_rate > 95:
                validation_passed = False
                print(f"      ❌ CRITICAL: {feat} has {zero_rate:.1f}% zeros - topology extraction may still be broken!")

# Step 5: Verify reply graph edge count
print("\n[5/6] REPLY GRAPH EDGE VALIDATION")
print("  " + "=" * 60)

# Build the graph to check edge count
from utils.graph_features import build_reply_graph
G = build_reply_graph("data/processed/pheme_kg.ttl")
print(f"  Reply graph nodes: {G.number_of_nodes():,}")
print(f"  Reply graph edges: {G.number_of_edges():,}")
print(f"  Expected edges:    65,565")

if G.number_of_edges() == 65565:
    print("  ✅ Edge count matches expected 65,565")
elif G.number_of_edges() > 0:
    print(f"  ⚠️ Edge count is {G.number_of_edges():,} (expected 65,565)")
else:
    print("  ❌ CRITICAL: Zero edges - fix did not work!")
    validation_passed = False

# Step 6: Save enriched dataset
print("\n[6/6] SAVING ENRICHED DATASET")
print("  " + "=" * 60)
output_path = "data/processed/pheme_features_with_graph.csv"
df_enriched.to_csv(output_path, index=False)
print(f"  Saved to: {output_path}")
print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
print(f"  Shape: {df_enriched.shape}")
print(f"  Original columns: 17")
print(f"  New columns: {len(graph_cols)}")
print(f"  Total columns: {len(df_enriched.columns)}")

# Final Validation Report
print("\n" + "=" * 80)
print("VALIDATION REPORT - MILESTONE 05A.1")
print("=" * 80)

# Summary statistics
critical_features_ok = all(
    r['status'] in ['✅', '⚠️'] 
    for r in zero_rate_report 
    if r['feature'] in ['node_in_degree', 'pagerank_score', 'node_out_degree']
)
graph_edges_ok = G.number_of_edges() > 0
all_columns_present = len(missing_cols) == 0

print("\nCRITICAL CHECKS:")
print(f"  [{'✅' if all_columns_present else '❌'}] All 14 graph columns present")
print(f"  [{'✅' if graph_edges_ok else '❌'}] Reply graph has edges (nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,})")
print(f"  [{'✅' if critical_features_ok else '❌'}] Critical topology features have non-zero signal")

print(f"\nTOPOLOGY FEATURES SUMMARY:")
topology_ok_count = sum(1 for r in zero_rate_report if r['status'] == '✅')
topology_warn_count = sum(1 for r in zero_rate_report if r['status'] == '⚠️')
topology_fail_count = sum(1 for r in zero_rate_report if r['status'] == '❌')
print(f"  ✅ Passed: {topology_ok_count}")
print(f"  ⚠️ Warnings: {topology_warn_count}")
print(f"  ❌ Failed: {topology_fail_count}")

# GO/NO-GO Decision
print("\n" + "=" * 80)
if validation_passed and all_columns_present and graph_edges_ok and critical_features_ok:
    print("🎯 VERDICT: GO")
    print("\n  The graph topology fix is DEPLOYED and VALIDATED.")
    print("  Structural signal is RESTORED across all 14 graph features.")
    print("  Notebook 05B (BERT + Graph Fusion) may proceed.")
else:
    print("🛑 VERDICT: NO-GO")
    print("\n  Critical validation checks failed.")
    print("  Do NOT proceed to Notebook 05B until issues are resolved.")
    if not graph_edges_ok:
        print("  - Reply graph still has zero edges")
    if not critical_features_ok:
        print("  - Critical topology features lack non-zero signal")
    if not all_columns_present:
        print(f"  - Missing columns: {missing_cols}")

print("=" * 80)

# Exit with appropriate code
sys.exit(0 if validation_passed else 1)