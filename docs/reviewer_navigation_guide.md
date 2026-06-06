# Thesis Reviewer Navigation Guide — Rumor Detection Project

**Date:** June 7, 2026  
**Purpose:** Help thesis reviewers and examiners navigate the repository and understand which notebooks correspond to which thesis chapters.

---

## Quick Start

```
Repository root: d:/rumor-detection-project/
Notebooks:       notebooks/
Documentation:   docs/
Results:         results/
Pipeline:        preprocessing/ → data/processed/ → models/
```

---

## Chapter-by-Chapter Notebook Mapping

### Chapter 2: Dataset Analysis

| Thesis Section | Notebook | Purpose | Key Outputs |
|---------------|----------|---------|-------------|
| 2.1 PHEME Dataset Overview | `notebooks/pheme_dataset_statistics.ipynb` | Comprehensive statistics of PHEME dataset (5 events, 5802 threads, 103K tweets) | Inline tables, `results/pheme_dataset_statistics.csv` |
| 2.1 Event Distribution | `notebooks/01_pheme_analysis.ipynb` | Exploratory analysis, event distribution, label distribution | Inline charts |
| 2.1 Summary Statistics | `notebooks/02_data_summary_statistics.ipynb` | Basic summary statistics (tweet counts, thread sizes) | Inline tables |

**Supporting files:**
- `results/pheme_dataset_statistics.csv` — 22 dataset metrics
- `results/pheme_dataset_statistics.md` — Formatted statistics report
- `notebooks/compute_pheme_statistics.py` — Standalone statistics generator

---

### Chapter 3: Baseline Models

| Thesis Section | Notebook | Purpose | Key Outputs |
|---------------|----------|---------|-------------|
| 3.1 Baseline Setup | `notebooks/03_rumor_detection_baseline_final_fixed.ipynb` | **Official baseline**: TF-IDF + Propagation + LogisticRegression | Final baseline metrics (Accuracy, Precision, Recall, F1) |
| — (History) | `notebooks/03_rumor_detection_baseline.ipynb` | Original baseline (event-based split) | Superseded |
| — (History) | `notebooks/03_rumor_detection_baseline_fixed.ipynb` | Fixed v1 baseline | Superseded |
| — (History) | `notebooks/03_rumor_detection_baseline_fixed_v2.ipynb` | Fixed v2 baseline | Superseded |

**Supporting scripts:**
- `first_baseline.py` — Earliest baseline implementation (event-based split)

---

### Chapter 4: Knowledge Graph & Graph Features

| Thesis Section | Notebook | Purpose | Key Outputs |
|---------------|----------|---------|-------------|
| 4.1 Ontology Design | `ontology/pheme_ontology_v1.ttl` | Ontology v1 (5 classes, 5 obj properties) | RDF/Turtle file |
| 4.2 Ontology v2 | `ontology/pheme_ontology_v2.ttl` | Ontology v2 (7 classes, 12 obj properties, 32 data properties) | RDF/Turtle file |
| 4.3 KG Construction | `knowledge_graph/build_kg.py` | KG v1 builder (RDF triples from pheme_features.csv) | `data/processed/pheme_kg.ttl` |
| 4.3 KG v2 | `knowledge_graph/build_kg_v2.py` | KG v2 builder (extended with user/event/thread properties) | `data/processed/pheme_kg_v2.ttl` |
| 4.4 Graph Feature Extraction | `src/extract_graph_features_v2.py` | 14 thread-level graph features from reply trees | `data/processed/graph_features_v2.csv` |
| 4.4 Graph Feature Pipeline | `utils/graph_features.py` | 14 post-level graph features (centrality, reputation, structure, authority) | `data/processed/pheme_features_with_graph.csv` |
| 4.5 Graph Feature Integration | `notebooks/04_graph_feature_integration_executed.ipynb` | **Official graph feature results**: TF-IDF + Graph + Propagation | Graph-enhanced metrics |
| 4.6 Topology Audit | `notebooks/05A_graph_topology_failure_audit.ipynb` | Forensic audit of URI parsing bug in KG topology | Bug validation report |
| 4.6 Topology Fix Validation | `validate_graph_fix.py` | Validates 14 graph features have non-zero topology signal | GO/NO-GO verdict |

**Supporting files:**
- `utils/visualization/plot_ontology_diagram.py` — Generates ontology diagram (Plot B)
- `utils/visualization/plot_class_hierarchy.py` — Generates class hierarchy (Plot A)
- `utils/visualization/plot_kg_sample.py` — Generates KG sample subgraph (Plot C)
- `utils/visualization/plot_kg_statistics.py` — Generates KG statistics (Plot D)
- `visualization/kg_semantic_final.png` — Publication-quality KG subgraph visualization

---

### Chapter 5: Hybrid Model (BERT + Graph Fusion)

| Thesis Section | Notebook | Purpose | Key Outputs |
|---------------|----------|---------|-------------|
| 5.1 Semantic Embeddings | `utils/bert_feature_fusion.py` | MiniLM sentence embeddings (384-dim), normalization, fusion matrix | In-memory feature matrices |
| 5.2 BERT + Graph Fusion | `notebooks/05_bert_graph_fusion.ipynb` | BERT (MiniLM) + Graph fusion experiments | Fusion metrics, FN analysis |
| 5.3 Full Hybrid Results | `notebooks/05_final_hybrid_results_thesis_executed.ipynb` | **PRIMARY RESULTS**: All 4 systems compared (TF-IDF, MiniLM, MiniLM+Graph, Hybrid) | `final_metrics_table.csv` |
| 5.3 Hybrid (fixed topology) | `notebooks/05B_bert_graph_fusion_fixed.py` | Post-fix fusion experiment (4 systems) | `regression_pack/05B_results.json` |
| 5.4 Ablation Study | `notebooks/ablation_study.ipynb` | 5-config ablation: TF-IDF → MiniLM → +Prop → +Graph → Full Hybrid | `results/ablation/ablation_table.csv`, 4 figures |
| — (Runner) | `notebooks/run_bert_graph_fusion.py` | Standalone experiment runner (6 systems) | `regression_pack/bert_graph_results.json` |
| — (Runner) | `run_ablation.py` | Standalone ablation study runner (5 configs) | `results/ablation/` |

---

### Chapter 8/9: Conclusion & Results Synthesis

| Thesis Section | Notebook | Purpose | Key Outputs |
|---------------|----------|---------|-------------|
| 8.1 Results Summary | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | **FINAL SYNTHESIS**: All models compared, 9 thesis figures | `notebooks/figures/fig_9_*.png` |
| 8.2 Performance Analysis | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | Recall progression, FN recovery, hard rumor analysis | 4 comparison figures |
| — (Final Metrics) | `final_metrics_table.csv` | **Official final metrics** (5 models, 7 metrics each) | Final thesis table |

---

## Figure Generation Reference

| Figure | Generated By | Output Path |
|--------|-------------|-------------|
| Ontology Class Hierarchy (Plot A) | `utils/visualization/plot_class_hierarchy.py` | `docs/figures/ontology_class_hierarchy.png` |
| Ontology Full Diagram (Plot B) | `utils/visualization/plot_ontology_diagram.py` | `docs/figures/ontology_full_diagram.png` |
| KG Sample Subgraph (Plot C) | `utils/visualization/plot_kg_sample.py` | `docs/figures/kg_sample_subgraph.png` |
| KG Statistics Chart (Plot D) | `utils/visualization/plot_kg_statistics.py` | `docs/figures/kg_statistics_chart.png` |
| KG Semantic Viz (thesis) | `notebooks/viz_kg_semantic_final.py` | `visualization/kg_semantic_final.png` |
| Metrics Comparison | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | `notebooks/figures/fig1_metrics_comparison.png` |
| FN Reduction | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | `notebooks/figures/fig2_fn_reduction.png` |
| Confusion Matrices | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | `notebooks/figures/fig3_confusion_matrices.png` |
| Ablation Study | `notebooks/ablation_study.ipynb` | `results/ablation/figures/fig1-4_*.png` |
| Ablation Study (alt) | `run_ablation.py` | `results/ablation/figures/fig1-4_*.png` |
| Master Performance | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | `notebooks/figures/fig_9_master_performance_comparison.png` |
| Recall Progression | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` | `notebooks/figures/fig_9_recall_progression_pipeline.png` |
| Ontology Mermaid Class | `generate_mermaid_diagram.py` | `ontology_mermaid_class_diagram.md` |
| Ontology Mermaid ER | `generate_mermaid_diagram.py` | `ontology_mermaid_er_diagram.md` |
| Ontology Graphviz | `generate_ontology_diagram.py` | `ontology_diagram` |

---

## Reproducing the Full Pipeline

To reproduce all results from scratch:

```bash
# 1. Preprocessing
python -m preprocessing.run_pipeline

# 2. Propagation Features
python -c "from preprocessing.propagation_features import *; main()"

# 3. KG Construction (v1)
python -m knowledge_graph.build_kg

# 4. Graph Features at post-level
python -c "from utils.graph_features import *; main()"

# 5. Graph Features at thread-level
python src/extract_graph_features_v2.py

# 6. KG v2 Construction
python -m knowledge_graph.build_kg_v2

# 7. Run notebooks in order:
#    notebooks/final_pipeline/ (proposed structure)
#    03_baseline_final_fixed → 04_graph_integration_executed
#    → 05_bert_graph_fusion → 05_hybrid_results_thesis_executed
#    → ablation_study → 09_results_synthesis_executed
```

---

## Quick Identification for Reviewers

| Reviewer Question | Answer |
|------------------|--------|
| "Where are the final numbers?" | `final_metrics_table.csv` |
| "Which notebook produced the final hybrid results?" | `notebooks/05_final_hybrid_results_thesis_executed.ipynb` |
| "Which notebook produced the thesis figures?" | `notebooks/09_final_thesis_results_synthesis_executed.ipynb` |
| "Where is the baseline comparison?" | `notebooks/03_rumor_detection_baseline_final_fixed.ipynb` |
| "Where is the ablation study?" | `notebooks/ablation_study.ipynb` or `run_ablation.py` |
| "How were graph features extracted?" | `utils/graph_features.py` and `src/extract_graph_features_v2.py` |
| "Where are the ontology files?" | `ontology/pheme_ontology_v1.ttl` and `v2.ttl` |
| "Where are the KG builders?" | `knowledge_graph/build_kg.py` and `build_kg_v2.py` |
| "How was the topology bug fixed?" | `notebooks/05A_graph_topology_failure_audit.ipynb` + `validate_graph_fix.py` |