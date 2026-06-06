# Notebook Reorganization Plan — Rumor Detection Project

**Date:** June 7, 2026  
**Purpose:** Propose a clean notebook directory structure for thesis reviewer readability.  
**Status:** PLAN ONLY — no files moved yet.

---

## Current Structure

```
notebooks/
├── 01_pheme_analysis.ipynb
├── 02_data_summary_statistics.html
├── 02_data_summary_statistics.ipynb
├── 03_rumor_detection_baseline.ipynb
├── 03_rumor_detection_baseline_final_fixed.ipynb
├── 03_rumor_detection_baseline_fixed.ipynb
├── 03_rumor_detection_baseline_fixed_v2.ipynb
├── 04_graph_feature_integration.ipynb
├── 04_graph_feature_integration_executed.ipynb
├── 05A_graph_topology_failure_audit.ipynb
├── 05B_bert_graph_fusion_fixed.py
├── 05_bert_graph_fusion.ipynb
├── 05_final_hybrid_results_thesis.ipynb
├── 05_final_hybrid_results_thesis_executed.ipynb
├── 09_final_thesis_results_synthesis.ipynb
├── 09_final_thesis_results_synthesis_executed.ipynb
├── ablation_study.ipynb
├── compute_pheme_statistics.py
├── final_metrics_table.csv
├── pheme_dataset_statistics.ipynb
├── quick_pheme_stats.py
├── run_bert_graph_fusion.py
├── viz_kg_semantic_final.py
└── figures/
    ├── fig1_metrics_comparison.png
    ├── fig2_fn_reduction.png
    ├── fig3_confusion_matrices.png
    ├── fig4_ablation_study.png
    ├── fig5_recovered_rumor_features.png
    ├── fig_9_false_negative_recovery.png
    ├── fig_9_hard_rumor_recovery.png
    ├── fig_9_master_performance_comparison.png
    └── fig_9_recall_progression_pipeline.png
```

---

## Proposed Structure

```
notebooks/
│
├── final_pipeline/                 ← Core notebooks for thesis results
│   ├── 03_baseline_final_fixed.ipynb
│   ├── 04_graph_integration_executed.ipynb
│   ├── 05_bert_graph_fusion.ipynb
│   ├── 05_hybrid_results_thesis_executed.ipynb
│   ├── 09_results_synthesis_executed.ipynb
│   └── ablation_study.ipynb
│
├── dataset_analysis/               ← Dataset exploration and statistics
│   ├── 01_pheme_analysis.ipynb
│   ├── 02_data_summary_statistics.ipynb
│   ├── 02_data_summary_statistics.html
│   └── pheme_dataset_statistics.ipynb
│
├── scripts/                        ← Standalone Python runners
│   ├── run_bert_graph_fusion.py
│   ├── 05B_bert_graph_fusion_fixed.py
│   ├── compute_pheme_statistics.py
│   ├── quick_pheme_stats.py
│   └── viz_kg_semantic_final.py
│
├── archived_history/               ← Research history (for reference)
│   ├── 03_rumor_detection_baseline.ipynb
│   ├── 03_rumor_detection_baseline_fixed.ipynb
│   ├── 03_rumor_detection_baseline_fixed_v2.ipynb
│   ├── 04_graph_feature_integration.ipynb
│   ├── 05_final_hybrid_results_thesis.ipynb
│   └── 09_final_thesis_results_synthesis.ipynb
│
├── figures/                        ← Generated thesis figures
│   └── (9 PNG files — unchanged)
│
└── final_metrics_table.csv         ← (keep or move to results/)
```

---

## Rationale

### Why `final_pipeline/`?
- Groups notebooks that produce the **final thesis results**.
- Reviewer can run these 6 notebooks in sequence to reproduce the entire experiment pipeline.
- Clear progression: Baseline → Graph Integration → BERT Fusion → Hybrid Results → Synthesis → Ablation.

### Why `dataset_analysis/`?
- Groups notebooks that explore and describe the PHEME dataset (Chapter 2).
- Separate from experimental notebooks for clarity.
- `02_data_summary_statistics.html` stays adjacent to its source notebook.

### Why `scripts/`?
- Standalone Python scripts are not notebooks but serve similar experimental purposes.
- Separating them from `.ipynb` files avoids confusion.

### Why `archived_history/`?
- Preserves research history without cluttering the primary view.
- Contains 6 notebooks that are either:
  - Superseded by final versions (3 baseline variants)
  - Unexecuted templates (04, 05, 09 unexecuted versions)
- Kept for reference/transparency but not needed for reproduction.

---

## Renaming Convention

Current long filenames (e.g. `03_rumor_detection_baseline_final_fixed.ipynb`) can be shortened in the final structure:

| Current Name | Proposed Short Name |
|---|---|
| `03_rumor_detection_baseline_final_fixed.ipynb` | `03_baseline_final_fixed.ipynb` |
| `04_graph_feature_integration_executed.ipynb` | `04_graph_integration_executed.ipynb` |
| `05_final_hybrid_results_thesis_executed.ipynb` | `05_hybrid_results_thesis_executed.ipynb` |
| `09_final_thesis_results_synthesis_executed.ipynb` | `09_results_synthesis_executed.ipynb` |

**However**, renaming may break internal references. If renaming is desired, a symlink or README mapping should be provided.

---

## Recommended Action

1. ✅ Create the 4 subdirectories
2. ✅ Move files according to plan
3. ✅ Update any internal import paths if needed
4. ✅ Create a `notebooks/README.md` explaining the structure

**Decision needed:** Should files be moved (restructuring) or should we use symlinks for backward compatibility?