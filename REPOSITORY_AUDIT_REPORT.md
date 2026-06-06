# REPOSITORY AUDIT REPORT

**Project:** Rumor Detection on Social Media using Ontology, Knowledge Graph and Machine Learning based on the PHEME Dataset  
**Date:** June 6, 2026  
**Auditor:** Repository Auditor

---

## 1. Repository Overview

| Metric | Value |
|--------|-------|
| **Total Folders** | 21 (excluding `.git` and `__pycache__`) |
| **Total Files (source/documentation)** | ~90 (excluding raw data JSON, `.git` internals, `.pyc` caches) |
| **Total Raw Data Files** | ~4,700 JSON files (PHEME dataset tweets) |
| **Major Project Modules** | 10 (preprocessing, ontology, knowledge_graph, models, utils, config, src, notebooks, visualization, project_brain_bundle) |

### High-Level Architecture (inferred from code)

```
Raw PHEME (JSON) 
    → preprocessing/ (loader → parser → normalizer → propagation_features)
    → data/processed/pheme_features.csv  
        → knowledge_graph/build_kg.py → pheme_kg.ttl (KG v1, Ontology v1)
        → knowledge_graph/build_kg_v2.py → pheme_kg_v2.ttl (KG v2, Ontology v2)
        → src/extract_graph_features_v2.py → graph_features_v2.csv
        → utils/graph_features.py → pheme_features_with_graph.csv
            → notebooks/ (03, 04, 05 series) 
            → utils/bert_feature_fusion.py (MiniLM embeddings)
                → final classification (LogisticRegression) 
                → final_metrics_table.csv / ablation results / figures
```

### Pipeline Flow
1. **Data Loading & Parsing:** PHEME raw JSON → structured CSV
2. **Feature Engineering:** Propagation features + Graph handcrafted features (14)
3. **Semantic Features:** TF-IDF (lexical) + MiniLM embeddings (semantic, 384-dim)
4. **Model Training:** LogisticRegression, Random Forest, XGBoost, LightGBM
5. **Evaluation:** Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MCC), Confusion Matrices
6. **Visualization:** Ontology diagrams, KG subgraph, statistics charts, ablation figures

---

## 2. Directory Inventory

| Directory | Purpose | Important Files |
|-----------|---------|-----------------|
| `config/` | Configuration (currently minimal) | `config.py` (placeholder) |
| `data/raw/pheme/` | Raw PHEME dataset (5 events) | ~4,700 JSON tweet files |
| `data/processed/` | Processed datasets | `pheme_clean.csv`, `pheme_features.csv`, `pheme_features_with_graph.csv`, `pheme_kg.ttl`, `pheme_kg_v2.ttl`, `graph_features_v2.csv` |
| `data/interim/` | Intermediate processing data | (empty) |
| `preprocessing/` | Data preprocessing pipeline | `loader.py`, `parser.py`, `normalizer.py`, `propagation_features.py`, `run_pipeline.py`, `features.py` |
| `ontology/` | Ontology definitions (OWL/TTL) | `pheme_ontology_v1.ttl`, `pheme_ontology_v2.ttl` |
| `knowledge_graph/` | Knowledge Graph construction | `build_kg.py` (KG v1), `build_kg_v2.py` (KG v2) |
| `models/` | ML model storage | `__init__.py` (placeholder) |
| `utils/` | Utility functions | `graph_features.py`, `bert_feature_fusion.py` |
| `utils/visualization/` | Figure generation scripts | `plot_class_hierarchy.py`, `plot_ontology_diagram.py`, `plot_kg_sample.py`, `plot_kg_statistics.py`, `run_all_figures.py` |
| `src/` | Additional source code | `extract_graph_features_v2.py` |
| `notebooks/` | Jupyter notebooks (analysis & experiments) | 14+ notebooks (01–09 series) |
| `notebooks/figures/` | Generated figures from notebooks | 5 figures (fig1–fig5) + 4 fig_9 variants |
| `project_brain_bundle/` | Internal development documentation | 12 markdown files + regression_pack/ |
| `results/` | Exported results & metrics | `pheme_dataset_statistics.csv`, `ablation_table.csv`, `pheme_dataset_statistics.md` |
| `results/ablation/figures/` | Ablation study figures | 4 figures (fig1–fig4) |
| `docs/` | Documentation | `VSCODE_EXTENSIONS_RECOMMENDED.md`, `figures/` |
| `visualization/` | Generated visualization outputs | `kg_semantic_final.png`, `kg_semantic_final.svg`, `kg_viz_log.txt` |
| `logs/` | Knowledge Graph build logs | `kg_build_after_fix.log`, `kg_build_v2.log` |
| `tests/` | Test files | (empty) |

---

## 3. File Inventory

### 3.1 Top-Level Files

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `main.py` | Python | Main entry point (prints help message only) | README |
| `README.md` | Markdown | Project overview and structure | — |
| `requirements.txt` | Text | Python dependencies (38 packages) | — |
| `.gitignore` | Git | Git ignore rules | — |
| `final_metrics_table.csv` | CSV | Final model comparison metrics (5 models) | thesis report, notebooks |
| `first_baseline.py` | Python | **First baseline**: LogisticRegression + TF-IDF + Propagation, event-based split | — |
| `run_ablation.py` | Python | **Ablation study runner**: 5 configs (TF-IDF → Full Hybrid), generates figures & tables | notebooks |
| `debug_graph_issue.py` | Python | Debug script: checks KG URI parsing for repliesTo edges | KG construction |
| `validate_graph_fix.py` | Python | **Validation script**: validates 14 graph features, zero-rate analysis, GO/NO-GO verdict | KG construction pipeline |
| `generate_mermaid_diagram.py` | Python | Generates Mermaid class and ER diagrams from ontology v1 TTL | Ontology visualization |
| `generate_ontology_diagram.py` | Python | Generates Graphviz diagram from ontology v1 TTL | Ontology visualization |
| `ontology_diagram` | File (no ext) | Graphviz-generated ontology diagram (likely PNG without extension) | Ontology visualization |
| `ontology_mermaid_class_diagram.md` | Markdown | Mermaid class diagram markdown | Ontology visualization |
| `ontology_mermaid_er_diagram.md` | Markdown | Mermaid ER diagram markdown | Ontology visualization |
| `project_survey_report.md` | Markdown | Comprehensive project survey report (Vietnamese) | Internal analysis |

### 3.2 Preprocessing Module (`preprocessing/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `preprocessing/__init__.py` | Python | Module init (stub) | — |
| `preprocessing/loader.py` | Python | **Data loader**: traverses PHEME dataset structure (events, labels, threads) | `run_pipeline.py` |
| `preprocessing/parser.py` | Python | **JSON parser**: reads source-tweet and reaction JSON files | `run_pipeline.py` |
| `preprocessing/normalizer.py` | Python | **Data normalizer**: converts raw tweets to structured schema | `run_pipeline.py` |
| `preprocessing/features.py` | Python | **Feature extraction stub** (TODO: not implemented) | — |
| `preprocessing/propagation_features.py` | Python | **Propagation features**: extracts node-level (depth, children_count) and thread-level (thread_size, reply_speed) features | Pipeline |
| `preprocessing/run_pipeline.py` | Python | **Pipeline orchestrator**: loads → parses → normalizes → exports CSV | `main.py` |

### 3.3 Ontology Module (`ontology/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `ontology/pheme_ontology_v1.ttl` | Turtle (RDF) | **Ontology v1**: 5 classes, 5 object properties, 8 data properties, 2 individuals, constraints | `build_kg.py`, `generate_mermaid_diagram.py`, `generate_ontology_diagram.py` |
| `ontology/pheme_ontology_v2.ttl` | Turtle (RDF) | **Ontology v2**: extends v1 with 2 subclasses (SourcePost, ReplyPost), 7 new object properties, 15 new data properties | `build_kg_v2.py` |

### 3.4 Knowledge Graph Module (`knowledge_graph/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `knowledge_graph/build_kg.py` | Python | **KG v1 builder**: constructs RDF triples from pheme_features.csv using ontology v1, includes cycle detection & validation | Pipeline |
| `knowledge_graph/build_kg_v2.py` | Python | **KG v2 builder**: extends v1 with ontology v2, user/event/thread properties, centrality scores, participatesInThread | Pipeline |

### 3.5 Core Utility Modules (`utils/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `utils/__init__.py` | Python | Module init (stub) | — |
| `utils/graph_features.py` | Python | **Graph feature extraction**: extracts 14 features (node centrality, user reputation, thread structure, source authority) from KG | `validate_graph_fix.py`, `run_bert_graph_fusion.py`, `05B_bert_graph_fusion_fixed.py` |
| `utils/bert_feature_fusion.py` | Python | **BERT fusion module**: MiniLM sentence embeddings (384-dim), feature normalization, fusion matrix construction | `run_bert_graph_fusion.py`, notebooks |
| `utils/visualization/__init__.py` | Python | Module init | — |
| `utils/visualization/plot_class_hierarchy.py` | Python | **Plot A**: Ontology class hierarchy diagram (v2) | Thesis figures |
| `utils/visualization/plot_ontology_diagram.py` | Python | **Plot B**: Full ontology diagram (v2 classes + object properties) | Thesis figures |
| `utils/visualization/plot_kg_sample.py` | Python | **Plot C**: KG sample subgraph (real conversation thread) | Thesis figures |
| `utils/visualization/plot_kg_statistics.py` | Python | **Plot D**: KG statistics chart (label distribution, event distribution, thread size) | Thesis figures |
| `utils/visualization/run_all_figures.py` | Python | Runner: executes all 4 plot scripts sequentially | Thesis figure generation |

### 3.6 Additional Source (`src/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `src/extract_graph_features_v2.py` | Python | **Thread-level graph feature extraction**: computes 14 features from reply tree (NetworkX) → `graph_features_v2.csv` | `run_ablation.py` |

### 3.7 Notebook Files (`notebooks/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `notebooks/01_pheme_analysis.ipynb` | Notebook | PHEME dataset analysis | Thesis |
| `notebooks/02_data_summary_statistics.ipynb` | Notebook | Data summary statistics + HTML report | Thesis |
| `notebooks/02_data_summary_statistics.html` | HTML | Exported HTML from notebook | — |
| `notebooks/03_rumor_detection_baseline.ipynb` | Notebook | Original baseline (TF-IDF + Propagation) | Thesis |
| `notebooks/03_rumor_detection_baseline_fixed.ipynb` | Notebook | Baseline fixed v1 | Thesis |
| `notebooks/03_rumor_detection_baseline_fixed_v2.ipynb` | Notebook | Baseline fixed v2 | Thesis |
| `notebooks/03_rumor_detection_baseline_final_fixed.ipynb` | Notebook | **Final baseline** | Thesis |
| `notebooks/04_graph_feature_integration.ipynb` | Notebook | Graph feature integration (unexecuted) | Thesis |
| `notebooks/04_graph_feature_integration_executed.ipynb` | Notebook | Graph feature integration (executed) | Thesis |
| `notebooks/05_bert_graph_fusion.ipynb` | Notebook | BERT + Graph fusion experiment | Thesis |
| `notebooks/05_final_hybrid_results_thesis.ipynb` | Notebook | Final hybrid results (unexecuted) | Thesis |
| `notebooks/05_final_hybrid_results_thesis_executed.ipynb` | Notebook | **Final hybrid results (executed)** | Thesis |
| `notebooks/05A_graph_topology_failure_audit.ipynb` | Notebook | Graph topology failure audit | Thesis |
| `notebooks/05B_bert_graph_fusion_fixed.py` | Python | Fixed BERT + Graph fusion (standalone Python script) | Thesis |
| `notebooks/09_final_thesis_results_synthesis.ipynb` | Notebook | Final results synthesis (unexecuted) | Thesis |
| `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | Notebook | **Final results synthesis (executed)** | Thesis |
| `notebooks/ablation_study.ipynb` | Notebook | Ablation study | Thesis |
| `notebooks/compute_pheme_statistics.py` | Python | Standalone PHEME statistics computation script | Results |
| `notebooks/quick_pheme_stats.py` | Python | Quick PHEME statistics (alternative to compute_pheme_statistics.py) | Results |
| `notebooks/run_bert_graph_fusion.py` | Python | BERT + Graph fusion experiment runner (standalone) | Results |
| `notebooks/viz_kg_semantic_final.py` | Python | **KG semantic visualization** (publication-quality) | `visualization/` outputs |
| `notebooks/final_metrics_table.csv` | CSV | Duplicate of root-level final_metrics_table.csv | Results |

### 3.8 Configuration (`config/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `config/config.py` | Python | Configuration placeholder (empty) | — |

### 3.9 Documentation (`docs/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `docs/VSCODE_EXTENSIONS_RECOMMENDED.md` | Markdown | VS Code extensions recommendations for the project | Developers |

### 3.10 Logs (`logs/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `logs/kg_build_after_fix.log` | Log | KG v1 build log (after fix) | Debugging |
| `logs/kg_build_v2.log` | Log | KG v2 build log | Debugging |
| `logs/__init__.py` | Python | Module init (has comment about knowledge graph) | — |

### 3.11 Results (`results/`)

| File Path | File Type | Purpose | Referenced By |
|-----------|-----------|---------|---------------|
| `results/pheme_dataset_statistics.csv` | CSV | PHEME dataset statistics (22 metrics) | Thesis |
| `results/pheme_dataset_statistics.md` | Markdown | PHEME statistics formatted as report | Thesis |
| `results/test.txt` | Text | Contains "hello" — test/placeholder file | — |
| `results/ablation/ablation_table.csv` | CSV | Ablation study metrics (5 configs) | Thesis |
| `results/ablation/ablation_table.md` | Markdown | Ablation study formatted as table | Thesis |

### 3.12 Project Brain Bundle (`project_brain_bundle/`)

| File Path | File Type | Purpose |
|-----------|-----------|---------|
| `project_brain_bundle/00_PROJECT_OVERVIEW.md` | Markdown | Project overview (internal) |
| `project_brain_bundle/01_CURRENT_PIPELINE_STATE.md` | Markdown | Current pipeline state |
| `project_brain_bundle/02_DATA_BEFORE_AFTER_FIX.md` | Markdown | Data comparison before/after fix |
| `project_brain_bundle/03_KNOWN_BUGS_AND_FIXES.md` | Markdown | Known bugs and fixes log |
| `project_brain_bundle/04_BASELINE_RESULTS.md` | Markdown | Baseline results documentation |
| `project_brain_bundle/05_GRAPH_INTEGRATION_PLAN.md` | Markdown | Graph integration plan |
| `project_brain_bundle/06_NEXT_ACTION_PROMPTS.md` | Markdown | Next action prompts |
| `project_brain_bundle/07_DO_NOT_BREAK_RULES.md` | Markdown | Do-not-break rules |
| `project_brain_bundle/08_GRAPH_FEATURE_RESULTS.md` | Markdown | Graph feature results |
| `project_brain_bundle/09_BERT_GRAPH_EXTENSION_PLAN.md` | Markdown | BERT extension plan |
| `project_brain_bundle/10_SESSION_HANDOFF_BERT_STAGE.md` | Markdown | Session handoff for BERT stage |
| `project_brain_bundle/11_CURRENT_CONTEXT.md` | Markdown | Current context snapshot |
| `project_brain_bundle/regression_pack/` | Directory | Regression results (various JSON, MD, CSV files) |

---

## 4. Ontology Assets

| File | Version | Purpose |
|------|---------|---------|
| `ontology/pheme_ontology_v1.ttl` | **v1** | Core ontology: 5 classes (Event, Post, User, ConversationThread, VeracityLabel), 5 object properties, 8 data properties. Used by KG v1 builder (`build_kg.py`). Referenced in thesis as initial ontology design. |
| `ontology/pheme_ontology_v2.ttl` | **v2** | Extended ontology: adds SourcePost/ReplyPost subclasses, 7 new object properties (hasSourcePost, hasReply, hasParentPost, hasChildPost, belongsToEvent, participatesInThread), 15 new data properties (postId, textLength, pagerankScore, userId, eventName, rumorLabel, etc.). Used by KG v2 builder (`build_kg_v2.py`). This is the final ontology described in the thesis. |

**Both ontology versions are explicitly required by the thesis** (Section: Ontology Engineering, which documents the evolution from v1 to v2).

---

## 5. Knowledge Graph Assets

| File | Version | Purpose |
|------|---------|---------|
| `knowledge_graph/build_kg.py` | **KG v1** | Builds RDF knowledge graph from `pheme_features.csv` using ontology v1. Output: `pheme_kg.ttl`. Includes: cycle detection, cross-thread validation, depth consistency checks. |
| `knowledge_graph/build_kg_v2.py` | **KG v2** | Builds extended RDF knowledge graph using ontology v2. Input: `pheme_features_with_graph.csv` (or fallback to `pheme_features.csv`). Output: `pheme_kg_v2.ttl`. Adds: user aggregates, thread-event mapping, source post metadata, centrality properties, participatesInThread relationships. |
| `data/processed/pheme_kg.ttl` | **KG v1** | Serialized RDF Turtle output from `build_kg.py`. Used by `utils/graph_features.py` to build NetworkX reply graph for feature extraction. |
| `data/processed/pheme_kg_v2.ttl` | **KG v2** | Serialized RDF Turtle output from `build_kg_v2.py`. |

**Both KG versions are explicitly required by the thesis** (Section: Knowledge Graph Construction, which documents the evolution from v1 to v2).

---

## 6. Feature Engineering Assets

### Semantic Features
| File | Feature Type | Purpose |
|------|-------------|---------|
| `utils/bert_feature_fusion.py` | **MiniLM embeddings (384-dim)** | Loads SentenceTransformer (`all-MiniLM-L6-v2`), extracts embeddings, normalizes, fuses with graph/propagation features. |
| `first_baseline.py` | **TF-IDF (5,000 features)** | Uses `TfidfVectorizer(max_features=5000)` for lexical features. |
| `run_ablation.py` | **TF-IDF (10,000 features)** | Uses `TfidfVectorizer(max_features=10000)` with n-gram range (1,2). |

### Propagation Features
| File | Feature Type | Purpose |
|------|-------------|---------|
| `preprocessing/propagation_features.py` | **Node-level**: depth, children_count, time_since_source, is_source | Extracted from thread reply tree structure. |
| `preprocessing/propagation_features.py` | **Thread-level**: thread_size, thread_duration_hours, first_reply_time_seconds, reply_speed_per_hour, max_depth | Aggregated at thread level for structural signals. |

### Graph Handcrafted Features
| File | Feature Type | Purpose |
|------|-------------|---------|
| `utils/graph_features.py` | **14 features** (4 groups): Node Centrality (5), User Reputation (4), Thread Structure (3), Source Authority (2) | Extracts from KG v1 (`pheme_kg.ttl`) using NetworkX. Output: `pheme_features_with_graph.csv`. |
| `src/extract_graph_features_v2.py` | **14 thread-level features**: thread_depth, num_nodes, num_edges, avg_branching_factor, max_branching_factor, source_reply_count, leaf_ratio, avg_depth, source_pagerank, avg_pagerank, source_centrality, avg_centrality, user_rumor_ratio, unique_users | Re-implements graph features at thread level from `pheme_features_with_graph.csv`. Output: `graph_features_v2.csv`. |

---

## 7. Model Training Assets

### Random Forest
| File | Purpose |
|------|---------|
| Referenced in thesis experiments. Used in notebooks (03–05 series). | Classification model for rumor detection. |

### XGBoost
| File | Purpose |
|------|---------|
| Referenced in thesis experiments. Used in notebooks. | Gradient boosting model for rumor detection. |

### LightGBM
| File | Purpose |
|------|---------|
| Referenced in thesis experiments. Used in notebooks. | Lightweight gradient boosting model. |

### Training/Evaluation Scripts
| File | Purpose |
|------|---------|
| `first_baseline.py` | Early baseline with LogisticRegression (event-based split). |
| `run_ablation.py` | Ablation study with LogisticRegression for 5 configs. |
| `notebooks/run_bert_graph_fusion.py` | BERT + Graph fusion experiments (6 systems). |
| `notebooks/05B_bert_graph_fusion_fixed.py` | Fixed topology experiment (4 systems: A/B/C/D). |
| `validate_graph_fix.py` | Graph fix validation + enriched dataset export. |

---

## 8. Experimental Result Assets

| File | Type | Content |
|------|------|---------|
| `final_metrics_table.csv` | CSV | Final metrics for 5 models (Acc, Prec, Recall, F1, ROC_AUC, MCC, FN). Best: Full Hybrid (Acc=0.9623, Recall=0.9606). |
| `notebooks/final_metrics_table.csv` | CSV | Duplicate of root-level metrics table. |
| `results/ablation/ablation_table.csv` | CSV | Ablation study: 5 configs with Delta metrics. |
| `results/ablation/ablation_table.md` | Markdown | Ablation results formatted. |
| `results/pheme_dataset_statistics.csv` | CSV | PHEME dataset statistics (22 rows). |
| `results/pheme_dataset_statistics.md` | Markdown | Dataset statistics with interpretation. |
| `notebooks/figures/fig1_metrics_comparison.png` | Figure | Metrics comparison bar chart. |
| `notebooks/figures/fig2_fn_reduction.png` | Figure | False negative reduction visualization. |
| `notebooks/figures/fig3_confusion_matrices.png` | Figure | Confusion matrices. |
| `notebooks/figures/fig4_ablation_study.png` | Figure | Ablation study visualization. |
| `notebooks/figures/fig5_recovered_rumor_features.png` | Figure | Recovered rumor features. |
| `notebooks/figures/fig_9_false_negative_recovery.png` | Figure | FN recovery for thesis. |
| `notebooks/figures/fig_9_hard_rumor_recovery.png` | Figure | Hard rumor recovery. |
| `notebooks/figures/fig_9_master_performance_comparison.png` | Figure | Master performance comparison. |
| `notebooks/figures/fig_9_recall_progression_pipeline.png` | Figure | Recall progression across pipeline. |
| `results/ablation/figures/fig1_ablation_bar_metrics.png` | Figure | Ablation bar metrics. |
| `results/ablation/figures/fig2_recall_fn_trend.png` | Figure | Recall/FN trend. |
| `results/ablation/figures/fig3_confusion_matrices.png` | Figure | Ablation confusion matrices. |
| `results/ablation/figures/fig4_feature_contribution.png` | Figure | Feature contribution analysis. |
| `visualization/kg_semantic_final.png` | Figure | KG semantic visualization (300 dpi). |
| `visualization/kg_semantic_final.svg` | Figure | KG semantic visualization (SVG vector). |

---

## 9. Documentation Assets

| File | Type | Purpose |
|------|------|---------|
| `README.md` | Project overview | High-level project description. |
| `project_survey_report.md` | Survey report | Comprehensive Vietnamese-language report covering all aspects of the project. |
| `docs/VSCODE_EXTENSIONS_RECOMMENDED.md` | Documentation | VS Code extension recommendations. |
| `ontology_mermaid_class_diagram.md` | Diagram | Mermaid class diagram for ontology. |
| `ontology_mermaid_er_diagram.md` | Diagram | Mermaid ER diagram for ontology. |
| `project_brain_bundle/` (12 files) | Internal docs | Session handoffs, pipeline states, progress tracking, known bugs, rules. |

---

## 10. Potentially Unrelated Files

| File | Reason It Might Be Unrelated | Confidence |
|------|-----------------------------|------------|
| `results/test.txt` | Contains only the word "hello". No relation to any thesis component. Appears to be a placeholder/test file. | **High** |
| `preprocessing/features.py` | Contains only `# TODO: Implement feature extraction functionality`. Empty stub, never implemented. Not referenced by any other code. | **High** |
| `config/config.py` | Contains only a single comment line (`# Configuration file for rumor detection project`). No actual configuration. Not referenced by any other code. | **High** |
| `models/__init__.py` | Contains only `# Models module for rumor detection`. Empty module, no models defined. Models are trained directly in notebooks/scripts, not stored as reusable modules. | **High** |
| `logs/__init__.py` | Contains `# Knowledge graph module for rumor detection` (mislabeled — it's in logs/ directory). Empty stub. | **Medium** |
| `logs/kg_build_after_fix.log` | Operational log file from KG v1 build. Useful for debugging but not directly referenced in thesis. | **Low-Medium** |
| `logs/kg_build_v2.log` | Operational log file from KG v2 build. Useful for debugging but not directly referenced in thesis. | **Low-Medium** |
| `debug_graph_issue.py` | Debugging script for KG URI parsing issue. Was used to diagnose a bug. Useful during development but not a thesis output. | **Low-Medium** |
| `project_survey_report.md` | Internal survey/report in Vietnamese. Contains valuable analysis but was not part of the final thesis submission. | **Low-Medium** |
| `project_brain_bundle/` (full directory) | Internal development documentation including session handoffs, prompts, and working notes. Not part of the thesis report. However, these were used to manage the development process. | **Low-Medium** |
| `notebooks/.ipynb_checkpoints/ablation_study-checkpoint.ipynb` | Jupyter notebook auto-save. Redundant with the main notebook. | **High** |
| `visualization/kg_viz_log.txt` | Debug log for the KG visualization script. Operational artifact. | **Low-Medium** |

---

## 11. Dependency Graph

```
Ontology (.ttl)
  ├── pheme_ontology_v1.ttl ──→ build_kg.py (KG v1) ──→ pheme_kg.ttl
  └── pheme_ontology_v2.ttl ──→ build_kg_v2.py (KG v2) ──→ pheme_kg_v2.ttl

KG Construction
  ├── KG v1 (pheme_kg.ttl) ──→ utils/graph_features.py ──→ pheme_features_with_graph.csv
  └── KG v2 (pheme_kg_v2.ttl) ──→ src/extract_graph_features_v2.py ──→ graph_features_v2.csv

Feature Extraction
  ├── Semantic (TF-IDF): first_baseline.py, run_ablation.py
  ├── Semantic (MiniLM): utils/bert_feature_fusion.py ──→ 384-dim embeddings
  ├── Propagation: preprocessing/propagation_features.py ──→ depth, thread_size, etc.
  └── Graph (14 features): utils/graph_features.py + src/extract_graph_features_v2.py

Model Training
  ├── first_baseline.py ──→ LogisticRegression (event split)
  ├── notebooks/03_* ──→ LogisticRegression, RF, XGBoost, LightGBM
  ├── notebooks/05_* ──→ MiniLM + Graph Fusion
  └── run_ablation.py ──→ Ablation study (5 configs)

Evaluation
  ├── final_metrics_table.csv ──→ 5 model comparison
  ├── results/ablation/ablation_table.csv ──→ ablation results
  └── notebooks/09_* ──→ Final synthesis

Figures/Tables
  ├── notebooks/figures/*.png ──→ Figures for thesis
  ├── results/ablation/figures/*.png ──→ Ablation figures
  ├── visualization/*.png, *.svg ──→ KG semantic visualization
  ├── ontology_mermaid_*.md ──→ Ontology Mermaid diagrams
  ├── ontology_diagram ──→ Graphviz ontology diagram
  └── utils/visualization/*.py ──→ Plot scripts (A–D) → docs/figures/
```

---

## 12. Thesis-Relevance Assessment

| Directory | Likely Needed For Thesis? | Reason |
|-----------|-------------------------|--------|
| `ontology/` | **Required** | Both v1 and v2 ontologies are explicitly documented in the thesis. |
| `knowledge_graph/` | **Required** | Both KG v1 and v2 builders are core to the thesis pipeline. |
| `preprocessing/` | **Required** | Data pipeline is essential for loading and processing PHEME dataset. |
| `utils/` | **Required** | Graph feature extraction and BERT fusion are critical for experiments. |
| `utils/visualization/` | **Required** | Generates thesis figures (A: class hierarchy, B: ontology diagram, C: KG subgraph, D: statistics). |
| `src/` | **Probably Required** | `extract_graph_features_v2.py` produces thread-level graph features used in ablation study. |
| `notebooks/` | **Required** | All notebooks (01–09) document the experimental evolution. Includes final executed results. |
| `results/` | **Required** | Contains final metrics, ablation results, dataset statistics. |
| `config/` | **Probably Not Required** | Empty placeholder file. |
| `models/` | **Probably Not Required** | Empty placeholder file. Models are trained in notebooks. |
| `logs/` | **Unclear** | Build logs are operational artifacts. May be useful for debugging reproducibility. |
| `project_brain_bundle/` | **Probably Not Required** | Internal development documentation. Not part of thesis report. |
| `docs/` | **Unclear** | VS Code extensions doc is a development helper, not thesis material. |
| `data/` | **Required** | Raw PHEME dataset and all processed CSVs/TTLs are essential for reproducibility. |
| `visualization/` | **Probably Required** | Contains the publication-quality KG semantic visualization used/usable in thesis. |
| `tests/` | **Probably Not Required** | Empty directory. |

---

## Summary

**Total files analyzed:** ~90 source/documentation files (excluding raw data JSON, `.git`, `__pycache__`)  
**Files likely needed for thesis:** ~75 (83%)  
**Files potentially unrelated (Low-Medium confidence):** ~12 (13%)  
**Files potentially unrelated (High confidence):** 5 (5%) — `results/test.txt`, `preprocessing/features.py`, `config/config.py`, `models/__init__.py`, `notebooks/.ipynb_checkpoints/ablation_study-checkpoint.ipynb`

**Key finding:** The vast majority of files are directly relevant to the thesis scope. The few files flagged with **High confidence** are empty stubs or placeholders. The `project_brain_bundle/` directory contains valuable development history but is not thesis material. The `logs/` directory contains operational artifacts from KG builds.

**No files should be deleted without explicit instruction.** The audit identifies potential candidates, but final decisions require thesis author review.