# Notebook Inventory — Rumor Detection Project

**Date:** June 7, 2026  
**Purpose:** Complete inventory of all files in `notebooks/` directory with classification and thesis relevance.

---

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| Jupyter Notebooks (.ipynb) | 15 | Interactive experiment notebooks |
| Python Scripts (.py) | 5 | Standalone experiment/utility scripts |
| HTML exports | 1 | Exported notebook output |
| CSV files | 1 | Metrics table (duplicate of root) |
| Figures directory | 1 | Contains 9 generated PNG figures |
| Total items | 23 | |

---

## Notebook Classification Legend

| Class | Meaning |
|-------|---------|
| **FINAL_CRITICAL** | Core notebook producing final thesis results. Must be kept. |
| **REPORT_SUPPORT** | Generates supporting data, statistics, or figures referenced in thesis. |
| **DEVELOPMENT_HISTORY** | Intermediate/incomplete notebook showing research evolution. Not in final thesis. |

---

## Full Inventory

### 1. `01_pheme_analysis.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | REPORT_SUPPORT |
| **Purpose** | Exploratory analysis of the PHEME dataset after preprocessing. Loads `pheme_clean.csv`, performs cleaning (duplicate removal), feature engineering (text_length, is_reply), and basic visualizations (label distribution, event distribution, thread sizes). |
| **Input** | `data/processed/pheme_clean.csv` |
| **Output** | In-memory statistics, inline charts (no exported files) |
| **Thesis Chapter** | Chapter 2 (Dataset Analysis) |
| **Status** | Executed cells visible (all 13 cells produce outputs) |
| **Notes** | Basic level analysis. Does not generate any exported figures. Could be regenerated from pipeline output. |

---

### 2. `02_data_summary_statistics.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | REPORT_SUPPORT |
| **Purpose** | Computes basic summary statistics of the PHEME dataset (tweet counts, event counts, thread counts, label distribution per event). |
| **Input** | `data/processed/pheme_clean.csv` |
| **Output** | Inline statistics only |
| **Thesis Chapter** | Chapter 2 (Dataset Analysis) |
| **Status** | Executed cells visible (all cells produce outputs) |
| **Notes** | Brief notebook (3 code cells). Simpler than `pheme_dataset_statistics.ipynb`. |

---

### 3. `02_data_summary_statistics.html`

| Attribute | Value |
|-----------|-------|
| **Classification** | REPORT_SUPPORT (exported copy) |
| **Purpose** | HTML export of `02_data_summary_statistics.ipynb` |
| **Input** | — |
| **Output** | — |
| **Thesis Chapter** | Chapter 2 |
| **Notes** | Duplicate of notebook in HTML format. Could be regenerated. |

---

### 4. `03_rumor_detection_baseline.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | DEVELOPMENT_HISTORY |
| **Purpose** | **Original baseline** using TfidfVectorizer + LogisticRegression with propagation features. Uses event-based split (train on 4 events, test on 1 event). |
| **Input** | `data/processed/pheme_features.csv` |
| **Output** | Baseline metrics (inline) |
| **Thesis Chapter** | Not directly referenced (superseded by final_fixed) |
| **Status** | Executed |
| **Notes** | Represents first baseline attempt. Superseded by `03_rumor_detection_baseline_final_fixed.ipynb`. Shows research evolution. |

---

### 5. `03_rumor_detection_baseline_fixed.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | DEVELOPMENT_HISTORY |
| **Purpose** | **First fixed version** of the baseline. Uses stratified train/test split (RANDOM_STATE=42) instead of event-based split. |
| **Input** | `data/processed/pheme_features.csv` |
| **Output** | Improved baseline metrics |
| **Thesis Chapter** | Not directly referenced (superseded by final_fixed) |
| **Status** | Executed |
| **Notes** | Intermediate improvement. Shows evolution of methodology. |

---

### 6. `03_rumor_detection_baseline_fixed_v2.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | DEVELOPMENT_HISTORY |
| **Purpose** | **Second fixed version** of the baseline. Further refinements to the pipeline. |
| **Input** | `data/processed/pheme_features.csv` |
| **Output** | Further improved baseline metrics |
| **Thesis Chapter** | Not directly referenced (superseded by final_fixed) |
| **Status** | Executed |
| **Notes** | Intermediate improvement. Shows evolution. |

---

### 7. `03_rumor_detection_baseline_final_fixed.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | **Final baseline experiment.** Uses TF-IDF + Propagation features + LogisticRegression with stratified split (RANDOM_STATE=42, test_size=0.2). Produces the official baseline metrics used in the thesis comparison. |
| **Input** | `data/processed/pheme_features.csv` |
| **Output** | Final baseline metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MCC, FN) |
| **Thesis Chapter** | Chapter 3 (Baseline Models) — **primary baseline results** |
| **Status** | Executed |
| **Notes** | This is the definitive baseline notebook. All other 03_* variants are history. |

---

### 8. `04_graph_feature_integration.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | DEVELOPMENT_HISTORY |
| **Purpose** | **Unexecuted** template for integrating graph features. Contains outline/plan for graph feature experiments. |
| **Input** | `data/processed/pheme_features_with_graph.csv` (intended) |
| **Output** | None (not executed) |
| **Thesis Chapter** | Not directly referenced |
| **Status** | **NOT executed** (no output cells) |
| **Notes** | Skeleton/predecessor of the executed version. Might represent the initial plan before execution. |

---

### 9. `04_graph_feature_integration_executed.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | **Executed version** of graph feature integration. Loads `pheme_features_with_graph.csv` (14 graph features), trains TF-IDF + Graph + Propagation models, compares performance gains. |
| **Input** | `data/processed/pheme_features_with_graph.csv` |
| **Output** | Graph-enhanced metrics, comparison tables |
| **Thesis Chapter** | Chapter 4 (Graph Feature Integration) — **primary results** |
| **Status** | Executed |
| **Notes** | This is the definitive graph feature integration notebook. |

---

### 10. `05_bert_graph_fusion.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | BERT (MiniLM) + Graph feature fusion experiments. Compares 4 systems: TF-IDF baseline, MiniLM only, MiniLM + Graph, Full Hybrid. |
| **Input** | `data/processed/pheme_features_with_graph.csv`, MiniLM model |
| **Output** | Fusion metrics, FN analysis, ablation comparison |
| **Thesis Chapter** | Chapter 5 (Hybrid Model) — **primary fusion results** |
| **Status** | Executed |
| **Notes** | Core hybrid experiment notebook. |

---

### 11. `05_final_hybrid_results_thesis.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | DEVELOPMENT_HISTORY |
| **Purpose** | **Unexecuted** template for thesis-ready hybrid results. Contains the structure but no computed outputs. |
| **Input** | (intended: pheme_features_with_graph.csv) |
| **Output** | None (not executed) |
| **Thesis Chapter** | Not directly referenced |
| **Status** | **NOT executed** |
| **Notes** | Unexecuted version of the final hybrid results. Predecessor to executed version. |

---

### 12. `05_final_hybrid_results_thesis_executed.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | **Executed thesis-ready hybrid results.** Full comparison of all systems (TF-IDF, MiniLM, Graph, Hybrid) with publication-quality figures and tables. |
| **Input** | `data/processed/pheme_features_with_graph.csv` |
| **Output** | Final metrics table (`final_metrics_table.csv`), comparison figures |
| **Thesis Chapter** | Chapter 5 (Hybrid Model) + Chapter 8 (Conclusion) — **primary thesis results** |
| **Status** | Executed |
| **Notes** | **Most important results notebook.** Contains 903 lines of thesis-ready output. |

---

### 13. `05A_graph_topology_failure_audit.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | Forensic audit of the "0 nodes/0 edges" bug in KG topology extraction. Proves the URI parsing bug and validates the fix. Produces GO/NO-GO verdict. |
| **Input** | `data/processed/pheme_kg.ttl` |
| **Output** | Bug validation report, edge count verification |
| **Thesis Chapter** | Chapter 4 (Methodology — debugging section) |
| **Status** | Executed |
| **Notes** | Important for demonstrating methodological rigor and bug discovery/fix process. |

---

### 14. `05B_bert_graph_fusion_fixed.py`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | Python script version of the fixed BERT + Graph fusion experiment. Runs 4 systems (A: TF-IDF, B: MiniLM, C: MiniLM+Graph, D: Hybrid) with validated topology. |
| **Input** | `data/processed/pheme_features_with_graph.csv` |
| **Output** | Results JSON (`project_brain_bundle/regression_pack/05B_results.json`) |
| **Thesis Chapter** | Chapter 5 (Hybrid Model — post-fix results) |
| **Status** | Standalone Python script (not a notebook) |
| **Notes** | Canonical experiment runner for post-fix results. Imports from `utils/bert_feature_fusion.py`. |

---

### 15. `09_final_thesis_results_synthesis.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | DEVELOPMENT_HISTORY |
| **Purpose** | **Unexecuted** template for final results synthesis. Intended to consolidate all prior results into thesis-ready format. |
| **Input** | (intended: various metrics CSVs) |
| **Output** | None (not executed) |
| **Thesis Chapter** | Intended for Chapter 8/9 (Conclusion + Results) |
| **Status** | **NOT executed** |
| **Notes** | Predecessor to executed version. |

---

### 16. `09_final_thesis_results_synthesis_executed.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | **Executed final results synthesis.** Consolidates all model results into 9 thesis-ready figures (fig_9_*) comparing baseline, graph, MiniLM, and hybrid performance. Includes recall progression, FN recovery analysis. |
| **Input** | `final_metrics_table.csv`, `results/ablation/ablation_table.csv` |
| **Output** | 9 figures (`notebooks/figures/fig_9_*.png`) |
| **Thesis Chapter** | Chapter 8 (Conclusion) + Chapter 9 (Results) — **final synthesis** |
| **Status** | Executed |
| **Notes** | **Second most important notebook.** Generates the figures used in thesis conclusion. |

---

### 17. `ablation_study.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | Comprehensive ablation study: 5 configs (TF-IDF → MiniLM → +Propagation → +Graph → Full Hybrid). Uses LogisticRegression with thread-level features from `graph_features_v2.csv`. Generates ablation tables and figures. |
| **Input** | `data/processed/pheme_features.csv`, `data/processed/graph_features_v2.csv` |
| **Output** | `results/ablation/ablation_table.csv`, `results/ablation/ablation_table.md`, `results/ablation/figures/*.png` (4 figures) |
| **Thesis Chapter** | Chapter 5 (Ablation Study section) |
| **Status** | Executed |
| **Notes** | 814 lines. Uses SentenceTransformer for MiniLM embeddings. Generates official ablation results. |

---

### 18. `compute_pheme_statistics.py`

| Attribute | Value |
|-----------|-------|
| **Classification** | REPORT_SUPPORT |
| **Purpose** | Standalone Python script that traverses raw PHEME JSON to compute dataset statistics. Outputs to `results/pheme_dataset_statistics.csv` and `results/pheme_dataset_statistics.md`. |
| **Input** | `data/raw/pheme/` (raw JSON) |
| **Output** | `results/pheme_dataset_statistics.csv`, `results/pheme_dataset_statistics.md` |
| **Thesis Chapter** | Chapter 2 (Dataset Statistics) |
| **Status** | Executable script (351 lines) |
| **Notes** | Comprehensive statistics computation from raw data. |

---

### 19. `pheme_dataset_statistics.ipynb`

| Attribute | Value |
|-----------|-------|
| **Classification** | REPORT_SUPPORT |
| **Purpose** | Notebook version of PHEME statistics. Also traverses raw JSON. Explicitly documents "used in Chapter 2 of the graduation thesis." |
| **Input** | `data/raw/pheme/` (raw JSON) |
| **Output** | Inline statistics + same CSV/MD outputs |
| **Thesis Chapter** | Chapter 2 (Dataset Analysis) — **explicitly referenced** |
| **Status** | Executed |
| **Notes** | 835 lines. Most comprehensive statistics notebook. **Explicitly says "used in Chapter 2" in its heading.** |

---

### 20. `quick_pheme_stats.py`

| Attribute | Value |
|-----------|-------|
| **Classification** | DEVELOPMENT_HISTORY |
| **Purpose** | Lightweight alternative to `compute_pheme_statistics.py`. Also traverses raw JSON but with simpler logic. Outputs the same CSV/MD files to `results/`. |
| **Input** | `data/raw/pheme/` (raw JSON) |
| **Output** | `results/pheme_dataset_statistics.csv`, `results/pheme_dataset_statistics.md` (same outputs) |
| **Thesis Chapter** | Not directly referenced |
| **Status** | Executable script (190 lines) |
| **Notes** | Alternative to `compute_pheme_statistics.py`. Both scripts produce the **exact same output files**, making this redundant. |

---

### 21. `run_bert_graph_fusion.py`

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** |
| **Purpose** | Standalone experiment runner for BERT + Graph fusion. Runs 6 systems (Baseline, Best Graph, BERT Only, BERT+Prop, BERT+Graph, Full Fusion). Saves results to `project_brain_bundle/regression_pack/bert_graph_results.json`. |
| **Input** | `data/processed/pheme_features.csv`, `data/processed/pheme_kg.ttl` |
| **Output** | `project_brain_bundle/regression_pack/bert_graph_results.json` |
| **Thesis Chapter** | Chapter 5 (Hybrid Model) |
| **Status** | Executable script (281 lines) |
| **Notes** | More comprehensive than notebook version. Imports from `utils/bert_feature_fusion.py`. |

---

### 22. `viz_kg_semantic_final.py`

| Attribute | Value |
|-----------|-------|
| **Classification** | REPORT_SUPPORT |
| **Purpose** | Publication-quality KG semantic visualization. Selects a sample thread from PHEME, builds an ontology graph with 7 relation types (belongsToEvent, hasSourcePost, etc.), and exports PNG + SVG. |
| **Input** | `data/processed/graph_features_v2.csv`, `data/processed/pheme_features_with_graph.csv` |
| **Output** | `visualization/kg_semantic_final.png`, `visualization/kg_semantic_final.svg`, `visualization/kg_viz_log.txt` |
| **Thesis Chapter** | Chapter 4 (Knowledge Graph Visualization) |
| **Status** | Executable script (376 lines) |
| **Notes** | Generates the publication-quality KG figure for the thesis. |

---

### 23. `figures/` (directory)

| Attribute | Value |
|-----------|-------|
| **Classification** | **FINAL_CRITICAL** (report assets) |
| **Purpose** | Contains 9 generated PNG figures used in the thesis: |
| | `fig1_metrics_comparison.png` — Metrics comparison bar chart |
| | `fig2_fn_reduction.png` — False negative reduction |
| | `fig3_confusion_matrices.png` — Confusion matrices |
| | `fig4_ablation_study.png` — Ablation study visualization |
| | `fig5_recovered_rumor_features.png` — Recovered rumor features |
| | `fig_9_false_negative_recovery.png` — FN recovery (conclusion) |
| | `fig_9_hard_rumor_recovery.png` — Hard rumor recovery |
| | `fig_9_master_performance_comparison.png` — Master comparison |
| | `fig_9_recall_progression_pipeline.png` — Recall progression |
| **Thesis Chapter** | Various chapters |
| **Status** | Generated artifacts |
| **Notes** | All figures can be regenerated from notebooks. |

---

## Classification Summary

| Class | Count | Files |
|-------|-------|-------|
| **FINAL_CRITICAL** | 8 | `03_rumor_detection_baseline_final_fixed.ipynb`, `04_graph_feature_integration_executed.ipynb`, `05_bert_graph_fusion.ipynb`, `05_final_hybrid_results_thesis_executed.ipynb`, `05A_graph_topology_failure_audit.ipynb`, `05B_bert_graph_fusion_fixed.py`, `09_final_thesis_results_synthesis_executed.ipynb`, `ablation_study.ipynb`, `run_bert_graph_fusion.py` |
| **REPORT_SUPPORT** | 8 | `01_pheme_analysis.ipynb`, `02_data_summary_statistics.ipynb`, `02_data_summary_statistics.html`, `pheme_dataset_statistics.ipynb`, `compute_pheme_statistics.py`, `viz_kg_semantic_final.py`, `04_graph_feature_integration.ipynb`, `05_final_hybrid_results_thesis.ipynb`, `09_final_thesis_results_synthesis.ipynb` |
| **DEVELOPMENT_HISTORY** | 5 | `03_rumor_detection_baseline.ipynb`, `03_rumor_detection_baseline_fixed.ipynb`, `03_rumor_detection_baseline_fixed_v2.ipynb`, `quick_pheme_stats.py` |
| **POSSIBLE_DUPLICATE** | 1 | `notebooks/final_metrics_table.csv` (duplicate of root) |