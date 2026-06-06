# Notebooks — Rumor Detection Project

**Project:** Phát hiện tin đồn trên mạng xã hội sử dụng Ontology và Knowledge Graph  
**Dataset:** PHEME (5 events, 5,802 threads, 102,440 tweets)  
**Author:** Phạm Phước Tài  
**Advisor:** TS. Phạm Thị Thu Thúy

---

## Repository Structure

```
notebooks/
├── archived_history/       ← Research history (7 files — not needed for reproduction)
├── figures/                ← Generated thesis figures (9 PNG files)
├── README.md               ← This file
├── ... (remaining notebooks and scripts)
```

---

## Notebook Pipeline (Suggested Execution Order)

For reviewers who want to reproduce the thesis results, follow this order:

### 1. Dataset Analysis (Chapter 2)
| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | `pheme_dataset_statistics.ipynb` | Comprehensive PHEME dataset statistics |
| 2 | `01_pheme_analysis.ipynb` | Exploratory data analysis |
| 3 | `02_data_summary_statistics.ipynb` | Basic summary statistics |

### 2. Baseline Models (Chapter 3)
| Order | Notebook | Purpose |
|-------|----------|---------|
| 4 | `03_rumor_detection_baseline_final_fixed.ipynb` | **Official baseline**: TF-IDF + Propagation |

### 3. Graph Feature Integration (Chapter 4)
| Order | Notebook | Purpose |
|-------|----------|---------|
| 5 | `04_graph_feature_integration_executed.ipynb` | **Graph feature results**: TF-IDF + Graph + Propagation |
| 6 | `05A_graph_topology_failure_audit.ipynb` | Topology bug audit (demonstrates methodological rigor) |

### 4. Hybrid Model — BERT + Graph Fusion (Chapter 5)
| Order | Notebook | Purpose |
|-------|----------|---------|
| 7 | `05_bert_graph_fusion.ipynb` | BERT (MiniLM) + Graph fusion experiments |
| 8 | `05_final_hybrid_results_thesis_executed.ipynb` | **PRIMARY RESULTS**: All systems compared |
| 9 | `ablation_study.ipynb` | Ablation study: 5 configs |

### 5. Conclusion & Results Synthesis (Chapter 8/9)
| Order | Notebook | Purpose |
|-------|----------|---------|
| 10 | `09_final_thesis_results_synthesis_executed.ipynb` | **FINAL SYNTHESIS**: 9 thesis-ready figures |

---

## Archived History

The following files have been moved to `archived_history/`:

| File | Reason |
|------|--------|
| `03_rumor_detection_baseline.ipynb` | Superseded by `final_fixed` |
| `03_rumor_detection_baseline_fixed.ipynb` | Superseded by `final_fixed` |
| `03_rumor_detection_baseline_fixed_v2.ipynb` | Superseded by `final_fixed` |
| `04_graph_feature_integration.ipynb` | Unexecuted template (executed version is canonical) |
| `05_final_hybrid_results_thesis.ipynb` | Unexecuted template (executed version is canonical) |
| `09_final_thesis_results_synthesis.ipynb` | Unexecuted template (executed version is canonical) |
| `quick_pheme_stats.py` | Redundant with `compute_pheme_statistics.py` |

These files are preserved for reference but are not needed to reproduce thesis results.

---

## Standalone Scripts

| Script | Purpose |
|--------|---------|
| `run_bert_graph_fusion.py` | BERT + Graph fusion experiment runner |
| `05B_bert_graph_fusion_fixed.py` | Post-fix fusion experiment (canonical) |
| `compute_pheme_statistics.py` | PHEME dataset statistics generator |
| `viz_kg_semantic_final.py` | Publication-quality KG visualization |

---

## Figure Reference

| Figure | Description |
|--------|-------------|
| `figures/fig1_metrics_comparison.png` | Metrics comparison bar chart |
| `figures/fig2_fn_reduction.png` | False negative reduction |
| `figures/fig3_confusion_matrices.png` | Confusion matrices |
| `figures/fig4_ablation_study.png` | Ablation study |
| `figures/fig5_recovered_rumor_features.png` | Recovered rumor features |
| `figures/fig_9_false_negative_recovery.png` | FN recovery (conclusion) |
| `figures/fig_9_hard_rumor_recovery.png` | Hard rumor recovery |
| `figures/fig_9_master_performance_comparison.png` | Master comparison |
| `figures/fig_9_recall_progression_pipeline.png` | Recall progression |

---

## Quick Navigation for Reviewers

| Question | Answer |
|----------|--------|
| "Where are the final metrics?" | `../final_metrics_table.csv` (root) |
| "Which notebook has the hybrid results?" | `05_final_hybrid_results_thesis_executed.ipynb` |
| "Which notebook has the thesis figures?" | `09_final_thesis_results_synthesis_executed.ipynb` |
| "Where is the baseline?" | `03_rumor_detection_baseline_final_fixed.ipynb` |
| "Where is the ablation study?" | `ablation_study.ipynb` |