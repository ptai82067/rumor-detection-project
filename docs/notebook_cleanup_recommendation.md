# Notebook Cleanup Recommendation — Rumor Detection Project

**Date:** June 7, 2026  
**Purpose:** Final classification of every notebook with recommended action.  
**Status:** RECOMMENDATION ONLY — no files moved or deleted.

---

## Classification Legend

| Tag | Meaning | Action |
|-----|---------|--------|
| **NEVER_DELETE** | Critical for thesis. Must be preserved permanently. | Keep in primary location. |
| **KEEP_AS_PRIMARY** | Active notebook used for reproduction or reference. | Keep accessible. |
| **ARCHIVE_HISTORY** | Shows research evolution. Not needed for reproduction. | Move to `archived_history/` subdirectory. |
| **POSSIBLE_DUPLICATE** | Redundant copy of another file. | Consider removing after verification. |

---

## Notebook Classifications

### FINAL_CRITICAL Notebooks (NEVER_DELETE)

| # | Notebook | Classification | Why |
|---|----------|---------------|-----|
| 1 | `03_rumor_detection_baseline_final_fixed.ipynb` | **NEVER_DELETE** | Produces official baseline metrics. Core Chapter 3 result. |
| 2 | `04_graph_feature_integration_executed.ipynb` | **NEVER_DELETE** | Produces official graph feature metrics. Core Chapter 4 result. |
| 3 | `05_bert_graph_fusion.ipynb` | **NEVER_DELETE** | Core BERT + Graph fusion experiment. Chapter 5. |
| 4 | `05_final_hybrid_results_thesis_executed.ipynb` | **NEVER_DELETE** | **Most important notebook.** Produces final hybrid results and metrics table. |
| 5 | `05A_graph_topology_failure_audit.ipynb` | **NEVER_DELETE** | Documents critical bug discovery/fix process. Demonstrates methodological rigor. |
| 6 | `09_final_thesis_results_synthesis_executed.ipynb` | **NEVER_DELETE** | **Second most important.** Generates 9 thesis-ready figures for conclusion. |
| 7 | `ablation_study.ipynb` | **NEVER_DELETE** | Produces official ablation results and 4 figures. |
| 8 | `run_bert_graph_fusion.py` | **NEVER_DELETE** | Standalone runner for BERT fusion experiment. |
| 9 | `05B_bert_graph_fusion_fixed.py` | **NEVER_DELETE** | Post-fix canonical experiment runner. |

### REPORT_SUPPORT Notebooks (KEEP_AS_PRIMARY)

| # | Notebook | Classification | Why |
|---|----------|---------------|-----|
| 10 | `01_pheme_analysis.ipynb` | **KEEP_AS_PRIMARY** | Supports Chapter 2 dataset analysis. |
| 11 | `02_data_summary_statistics.ipynb` | **KEEP_AS_PRIMARY** | Supports Chapter 2 summary statistics. |
| 12 | `02_data_summary_statistics.html` | **KEEP_AS_PRIMARY** | Convenience export for reviewers. |
| 13 | `pheme_dataset_statistics.ipynb` | **KEEP_AS_PRIMARY** | Explicitly says "used in Chapter 2" in heading. Most comprehensive statistics. |
| 14 | `compute_pheme_statistics.py` | **KEEP_AS_PRIMARY** | Canonical statistics generator script. |
| 15 | `viz_kg_semantic_final.py` | **KEEP_AS_PRIMARY** | Generates publication-quality KG figure for thesis. |

### DEVELOPMENT_HISTORY Notebooks (ARCHIVE_HISTORY)

| # | Notebook | Classification | Why |
|---|----------|---------------|-----|
| 16 | `03_rumor_detection_baseline.ipynb` | **ARCHIVE_HISTORY** | Original baseline (event split). Superseded by `final_fixed`. Confidence: 98%. |
| 17 | `03_rumor_detection_baseline_fixed.ipynb` | **ARCHIVE_HISTORY** | Fixed v1 (stratified split). Superseded by `final_fixed`. Confidence: 98%. |
| 18 | `03_rumor_detection_baseline_fixed_v2.ipynb` | **ARCHIVE_HISTORY** | Fixed v2. Superseded by `final_fixed`. Confidence: 98%. |
| 19 | `04_graph_feature_integration.ipynb` | **ARCHIVE_HISTORY** | Unexecuted template. Executed version is the canonical one. Confidence: 95%. |
| 20 | `05_final_hybrid_results_thesis.ipynb` | **ARCHIVE_HISTORY** | Unexecuted template. Executed version is the canonical one. Confidence: 95%. |
| 21 | `09_final_thesis_results_synthesis.ipynb` | **ARCHIVE_HISTORY** | Unexecuted template. Executed version is the canonical one. Confidence: 95%. |
| 22 | `quick_pheme_stats.py` | **ARCHIVE_HISTORY** | Simplified alternative to `compute_pheme_statistics.py`. Both produce same output. Confidence: 98%. |

### POSSIBLE_DUPLICATE Files

| # | File | Classification | Why |
|---|------|---------------|-----|
| 23 | `notebooks/final_metrics_table.csv` | **POSSIBLE_DUPLICATE** | Exact copy of root `final_metrics_table.csv`. Kept for notebook convenience. Confidence: 100%. |

---

## Summary of Recommendations

| Action | Count | Files |
|--------|-------|-------|
| **NEVER_DELETE** | 9 | All FINAL_CRITICAL notebooks |
| **KEEP_AS_PRIMARY** | 6 | All REPORT_SUPPORT notebooks |
| **ARCHIVE_HISTORY** | 7 | 3 baseline variants + 3 unexecuted templates + 1 redundant script |
| **POSSIBLE_DUPLICATE** | 1 | `notebooks/final_metrics_table.csv` |

---

## Proposed Actions

### Action 1: Archive Development History (7 files)
Move the following to `notebooks/archived_history/`:
- `03_rumor_detection_baseline.ipynb`
- `03_rumor_detection_baseline_fixed.ipynb`
- `03_rumor_detection_baseline_fixed_v2.ipynb`
- `04_graph_feature_integration.ipynb`
- `05_final_hybrid_results_thesis.ipynb`
- `09_final_thesis_results_synthesis.ipynb`
- `quick_pheme_stats.py`

**Impact:** These files remain in the repository but are moved to a subdirectory. They are still accessible for reference. No data loss.

### Action 2: Keep all other files in current locations
All FINAL_CRITICAL, REPORT_SUPPORT, and POSSIBLE_DUPLICATE files stay in place.

### Action 3: Create notebooks/README.md
Add a README explaining the directory structure to help reviewers navigate.

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Archived notebook still imported by another script | **Very Low** | All archived notebooks are self-contained. No cross-notebook imports. |
| Reviewer wants to see development history | **Low** | Files remain in repository under `archived_history/`. |
| Unexecuted template has unique content not in executed version | **Very Low** | Executed versions contain all code from templates plus computed outputs. |

## Final Verification Statement

After implementing the archive actions, the project will still:

1. ✅ Reproduce all thesis results (all FINAL_CRITICAL notebooks intact)
2. ✅ Generate all thesis figures (all figure generators intact)
3. ✅ Navigate clearly by chapter (thanks to reorganization)
4. ✅ Preserve development history (archived, not deleted)
5. ✅ Support future UI development (all core modules intact)